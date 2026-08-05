/*
Copyright (C) 2026, Q2RTX contributors.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

/*
	Training-data capture ("capture")
	=================================

	Dumps square ground-truth tiles out of the rendered frame, for training the
	NPU upscaler (upscaler.c) on in-domain Quake 2 RTX imagery rather than on the
	DIV2K photographs its stock weights come from.

	The source is IMG_TAA_OUTPUT read *after* tone mapping, which is exactly
	where upscaler_pack.comp samples: no HUD, view weapon included, same content
	the model sees at runtime. Note that the older VKPT_IMAGE_DUMPS hook in
	main.c copies the same image *before* tone mapping, so it is not equivalent.

	Tiles are written as 16-bit PNGs holding the tone-mapped *linear* values,
	i.e. the encode in shader/upscaler_shared.h (pow(1/2.2)) is deliberately
	*not* applied here. That is left to the offline dataset step
	(scripts/upscaler_dataset.py), so the transfer curve can be retuned without
	recapturing anything. Low-resolution training inputs are produced there too,
	by downscaling these tiles.

	This is not gated on USE_ORT_QNN_UPSCALER: capture has to run on the
	machines the data is collected on, which are not the Windows-ARM64 machines
	the model is deployed to.

	Q2RTX cvars
	-----------
	* capture_tile            - tile edge in pixels (512, the model's output edge).
	* capture_tiles_per_frame - tiles written per captured frame.
	* capture_min_stddev      - reject tiles whose luminance standard deviation is
	                            below this, which throws out flat sky and unlit walls.
	* capture_tag             - filename prefix; the roam bot (client/capture_bot.c)
	                            sets it to the current map name.
	* capture_prefix          - extra filename prefix identifying the collection
	                            run, so several passes accumulate into one folder
	                            instead of overwriting each other.

	Console commands
	----------------
	* capture_shot - write capture_tiles_per_frame tiles from the next rendered
	  frame into <gamedir>/screenshots/dataset/, plus one line per tile in
	  manifest.jsonl recording where the camera was.
*/

#include "shared/shared.h"
#include "common/common.h"
#include "vkpt.h"
#include "vk_util.h"
#include "conversion.h"

#include <zlib.h>

// Bound on the tile edge, so a fat-fingered cvar cannot ask for a multi-gigabyte
// scratch allocation.
#define CAPTURE_MAX_TILE     4096
// Upper bound on tiles taken from one frame. A 3840x2160 frame holds 28 disjoint
// 512 tiles, so this leaves room for smaller tiles or larger frames.
#define CAPTURE_MAX_TILES    128

static cvar_t *cvar_capture_tile = NULL;
static cvar_t *cvar_capture_tiles_per_frame = NULL;
static cvar_t *cvar_capture_min_stddev = NULL;
static cvar_t *cvar_capture_tag = NULL;
static cvar_t *cvar_capture_prefix = NULL;

static struct {
	// One-shot flag set by the capture_shot command, consumed by
	// vkpt_capture_record() and cleared by vkpt_capture_writeout().
	bool         requested;
	// Set by vkpt_capture_record() once the copy is actually in the command
	// buffer, so writeout can tell a real capture from a frame where the pass
	// bailed out and the readback image may not even exist.
	bool         recorded;

	// Rect of the readback image that vkpt_capture_record() filled, which is
	// the part of IMG_TAA_OUTPUT that vkpt_taa() actually wrote this frame.
	VkExtent2D   recorded_extent;

	// Camera for the recorded frame, carried through to the manifest.
	vec3_t       vieworg;
	vec3_t       viewangles;
	uint64_t     frame;

	// Host-visible linear copy of the tone-mapped frame. Allocated on the first
	// capture and resized whenever the render extent changes, so that a session
	// that never captures anything pays nothing.
	VkImage        image;
	VkDeviceMemory memory;
	VkDeviceSize   memory_size;
	VkExtent2D     image_extent;

	unsigned     tiles_written;
} capture;

/*
================================================================================
16-bit PNG writer

stb_image_write only emits 8 bits per component, and the whole point of dumping
linear values is to keep the precision that an 8-bit encode would throw away, so
the PNG is assembled here directly on top of zlib (which the engine already
links for .pkz and demo compression).
================================================================================
*/

static void png_put32(byte *p, uint32_t v)
{
	p[0] = (byte)(v >> 24);
	p[1] = (byte)(v >> 16);
	p[2] = (byte)(v >> 8);
	p[3] = (byte)v;
}

// Appends one PNG chunk (length, type, payload, CRC over type+payload).
static byte *png_chunk(byte *out, const char *type, const byte *data, size_t length)
{
	png_put32(out, (uint32_t)length);
	out += 4;

	memcpy(out, type, 4);
	memcpy(out + 4, data, length);

	uLong crc = crc32(0, out, (uInt)(4 + length));
	out += 4 + length;

	png_put32(out, (uint32_t)crc);
	return out + 4;
}

// Writes an 8-bit-per-sample-pair RGB image. 'samples' holds width * height * 3
// big-endian uint16 values already in PNG sample order.
static bool write_png16(const char *path, const byte *samples, uint32_t width, uint32_t height)
{
	static const byte signature[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

	bool   ok = false;
	byte  *raw = NULL, *compressed = NULL, *file = NULL;

	// Scanlines, each prefixed with a filter-type byte. Filter 0 (none) keeps
	// this simple; deflate still gets most of the win on flat regions, and these
	// tiles are noisy enough that the smarter filters buy little.
	size_t row_bytes = (size_t)width * 3 * 2;
	size_t raw_size = (row_bytes + 1) * height;

	raw = Z_Malloc(raw_size);
	for (uint32_t y = 0; y < height; y++) {
		byte *dst = raw + (row_bytes + 1) * y;
		*dst++ = 0;
		memcpy(dst, samples + row_bytes * y, row_bytes);
	}

	uLongf compressed_size = compressBound((uLong)raw_size);
	compressed = Z_Malloc(compressed_size);

	if (compress2(compressed, &compressed_size, raw, (uLong)raw_size, Z_DEFAULT_COMPRESSION) != Z_OK) {
		Com_EPrintf("capture: deflate failed for '%s'\n", path);
		goto done;
	}

	{
		byte ihdr[13];
		png_put32(ihdr + 0, width);
		png_put32(ihdr + 4, height);
		ihdr[8]  = 16; // bit depth
		ihdr[9]  = 2;  // color type: truecolor RGB
		ihdr[10] = 0;  // deflate
		ihdr[11] = 0;  // adaptive filtering
		ihdr[12] = 0;  // no interlace

		// signature + three chunks; 12 bytes of framing per chunk.
		size_t file_size = sizeof(signature) + (12 + sizeof(ihdr)) + (12 + compressed_size) + 12;
		file = Z_Malloc(file_size);

		byte *p = file;
		memcpy(p, signature, sizeof(signature));
		p += sizeof(signature);

		p = png_chunk(p, "IHDR", ihdr, sizeof(ihdr));
		p = png_chunk(p, "IDAT", compressed, compressed_size);
		p = png_chunk(p, "IEND", NULL, 0);

		// Written through stdio rather than the FS layer: these paths are
		// absolute OS paths under fs_gamedir, the same way upscaler.c writes its
		// tensor dumps.
		FILE *fp = fopen(path, "wb");
		if (!fp) {
			Com_EPrintf("capture: failed to open '%s' for writing\n", path);
			goto done;
		}

		size_t file_bytes = (size_t)(p - file);
		ok = fwrite(file, 1, file_bytes, fp) == file_bytes;
		fclose(fp);

		if (!ok)
			Com_EPrintf("capture: short write to '%s'\n", path);
	}

done:
	Z_Free(file);
	Z_Free(compressed);
	Z_Free(raw);
	return ok;
}

/*
================================================================================
Readback image
================================================================================
*/

static void destroy_readback(void)
{
	if (!capture.image)
		return;

	vkDestroyImage(qvk.device, capture.image, NULL);
	vkFreeMemory(qvk.device, capture.memory, NULL);

	capture.image = VK_NULL_HANDLE;
	capture.memory = VK_NULL_HANDLE;
	capture.memory_size = 0;
	capture.image_extent.width = 0;
	capture.image_extent.height = 0;
}

// Creates (or resizes) the host-visible copy target and leaves it in GENERAL,
// which is the layout vkpt_capture_record() expects to transition away from.
static bool create_readback(VkExtent2D extent)
{
	if (capture.image &&
		capture.image_extent.width == extent.width &&
		capture.image_extent.height == extent.height)
	{
		return true;
	}

	// Previous dispatches may still be reading it, and the copy is recorded into
	// a command buffer that is submitted later in the frame.
	vkDeviceWaitIdle(qvk.device);
	destroy_readback();

	VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R16G16B16A16_SFLOAT, // matches IMG_TAA_OUTPUT
		.extent.width = extent.width,
		.extent.height = extent.height,
		.extent.depth = 1,
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_LINEAR,
		.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	if (vkCreateImage(qvk.device, &image_info, NULL, &capture.image) != VK_SUCCESS) {
		Com_EPrintf("capture: failed to create a %ux%u readback image\n", extent.width, extent.height);
		capture.image = VK_NULL_HANDLE;
		return false;
	}

	VkMemoryRequirements mem_req;
	vkGetImageMemoryRequirements(qvk.device, capture.image, &mem_req);

	VkMemoryAllocateInfo mem_alloc_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_req.size,
		.memoryTypeIndex = get_memory_type(mem_req.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
	};

	if (vkAllocateMemory(qvk.device, &mem_alloc_info, NULL, &capture.memory) != VK_SUCCESS ||
		vkBindImageMemory(qvk.device, capture.image, capture.memory, 0) != VK_SUCCESS)
	{
		Com_EPrintf("capture: failed to allocate %zu bytes for the readback image\n", (size_t)mem_req.size);
		destroy_readback();
		return false;
	}

	capture.memory_size = mem_req.size;
	capture.image_extent = extent;

	VkCommandBuffer cmd_buf = vkpt_begin_command_buffer(&qvk.cmd_buffers_graphics);

	IMAGE_BARRIER_STAGES(cmd_buf,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VK_PIPELINE_STAGE_HOST_BIT,
		.image = capture.image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
	);

	vkpt_submit_command_buffer_simple(cmd_buf, qvk.queue_graphics, false);
	vkpt_wait_idle(qvk.queue_graphics, &qvk.cmd_buffers_graphics);

	Com_DPrintf("capture: readback image is %ux%u (%zu KiB)\n",
		extent.width, extent.height, (size_t)(mem_req.size / 1024));
	return true;
}

/*
================================================================================
Tile selection and writeout
================================================================================
*/

// Rec. 709 luminance of one texel of the mapped readback image.
static float tile_luma(const byte *row_base, uint32_t x)
{
	const uint16_t *texel = (const uint16_t *)row_base + (size_t)x * 4;

	float r = halfToFloat(texel[0]);
	float g = halfToFloat(texel[1]);
	float b = halfToFloat(texel[2]);

	return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Standard deviation of luminance over the tile, on a coarse grid -- this only
// has to separate "flat sky" from "has structure", and sampling every 8th texel
// makes the rejection loop cheap enough to run several times per slot.
static float tile_stddev(const byte *data, VkDeviceSize row_pitch, uint32_t x0, uint32_t y0, uint32_t tile)
{
	const uint32_t step = max(1u, tile / 32);

	double sum = 0.0, sum_sq = 0.0;
	unsigned count = 0;

	for (uint32_t y = 0; y < tile; y += step) {
		const byte *row = data + row_pitch * (y0 + y);
		for (uint32_t x = 0; x < tile; x += step) {
			float luma = tile_luma(row, x0 + x);
			sum += luma;
			sum_sq += (double)luma * luma;
			count++;
		}
	}

	if (!count)
		return 0.f;

	double mean = sum / count;
	double variance = sum_sq / count - mean * mean;
	return variance > 0.0 ? (float)sqrt(variance) : 0.f;
}

static void write_manifest_line(const char *tag, uint32_t x, uint32_t y, uint32_t tile)
{
	char path[MAX_OSPATH];
	if (Q_snprintf(path, sizeof(path), "%s/screenshots/dataset/manifest.jsonl", fs_gamedir) >= sizeof(path))
		return;

	FILE *fp = fopen(path, "ab");
	if (!fp)
		return;

	char line[512];
	size_t len = Q_snprintf(line, sizeof(line),
		"{\"prefix\":\"%s\",\"tag\":\"%s\",\"frame\":%" PRIu64 ",\"x\":%u,\"y\":%u,\"tile\":%u,"
		"\"render_w\":%u,\"render_h\":%u,"
		"\"origin\":[%.2f,%.2f,%.2f],\"angles\":[%.2f,%.2f,%.2f]}\n",
		cvar_capture_prefix->string, tag, capture.frame, x, y, tile,
		capture.recorded_extent.width, capture.recorded_extent.height,
		capture.vieworg[0], capture.vieworg[1], capture.vieworg[2],
		capture.viewangles[0], capture.viewangles[1], capture.viewangles[2]);

	if (len < sizeof(line))
		fwrite(line, 1, len, fp);

	fclose(fp);
}

// Converts one tile of the mapped fp16 image into big-endian uint16 RGB and
// writes it. The values stay linear: clamping to [0,1] is the only transform,
// and it matches the clamp upscaler_linear_to_encoded() applies at runtime, so
// nothing is discarded that the model could ever have seen.
static bool write_tile(const byte *data, VkDeviceSize row_pitch, uint32_t x0, uint32_t y0, uint32_t tile,
	const char *tag)
{
	byte *samples = Z_Malloc((size_t)tile * tile * 3 * 2);

	for (uint32_t y = 0; y < tile; y++) {
		const uint16_t *src = (const uint16_t *)(data + row_pitch * (y0 + y)) + (size_t)x0 * 4;
		byte *dst = samples + (size_t)y * tile * 3 * 2;

		for (uint32_t x = 0; x < tile; x++) {
			for (int c = 0; c < 3; c++) {
				float value = halfToFloat(src[c]);
				value = Q_clipf(value, 0.f, 1.f);

				uint16_t quantized = (uint16_t)(value * 65535.f + 0.5f);
				*dst++ = (byte)(quantized >> 8);
				*dst++ = (byte)quantized;
			}
			src += 4;
		}
	}

	char path[MAX_OSPATH];
	bool ok = false;

	if (Q_snprintf(path, sizeof(path), "%s/screenshots/dataset/%s%s_%" PRIu64 "_%u_%u.png",
			fs_gamedir, cvar_capture_prefix->string, tag, capture.frame, x0, y0) >= sizeof(path))
	{
		Com_EPrintf("capture: dump path too long\n");
		goto done;
	}

	if (FS_CreatePath(path) < 0) {
		Com_EPrintf("capture: failed to create directory for '%s'\n", path);
		goto done;
	}

	ok = write_png16(path, samples, tile, tile);
	if (ok)
		write_manifest_line(tag, x0, y0, tile);

done:
	Z_Free(samples);
	return ok;
}

static void Capture_Shot_f(void)
{
	capture.requested = true;
}

static const cmdreg_t capture_cmds[] = {
	{ "capture_shot", &Capture_Shot_f, NULL },
	{ NULL, NULL, NULL }
};

/*
================================================================================
Public interface
================================================================================
*/

void vkpt_capture_init_cvars(void)
{
	// Defaults describe the QuickSRNet tile: 512 out, which the offline step
	// downscales to the model's 128 input.
	cvar_capture_tile = Cvar_Get("capture_tile", "512", CVAR_ARCHIVE);
	cvar_capture_tiles_per_frame = Cvar_Get("capture_tiles_per_frame", "4", CVAR_ARCHIVE);
	cvar_capture_min_stddev = Cvar_Get("capture_min_stddev", "0.02", CVAR_ARCHIVE);
	cvar_capture_tag = Cvar_Get("capture_tag", "capture", 0);
	// Distinguishes one collection run from the next. Tile names are built from
	// the map and the frame counter, and the frame counter restarts every run,
	// so without this a second pass silently overwrites the first instead of
	// adding to it.
	cvar_capture_prefix = Cvar_Get("capture_prefix", "", 0);

	Cmd_Register(capture_cmds);
}

void vkpt_capture_shutdown(void)
{
	destroy_readback();
	capture.requested = false;
	capture.recorded = false;
}

bool vkpt_capture_pending(void)
{
	return capture.requested;
}

void vkpt_capture_record(VkCommandBuffer cmd_buf)
{
	capture.recorded = false;

	if (!capture.requested)
		return;

	VkExtent2D extent = qvk.extent_taa_output;
	if (!extent.width || !extent.height) {
		capture.requested = false;
		return;
	}

	int tile = Q_clip(cvar_capture_tile->integer, 16, CAPTURE_MAX_TILE);
	if ((uint32_t)tile > extent.width || (uint32_t)tile > extent.height) {
		Com_EPrintf("capture: the render extent is %ux%u, which cannot hold a %dx%d tile; "
			"raise the resolution or lower capture_tile\n", extent.width, extent.height, tile, tile);
		capture.requested = false;
		return;
	}

	if (!create_readback(extent)) {
		capture.requested = false;
		return;
	}

	VkImage src_image = qvk.images[VKPT_IMG_TAA_OUTPUT];

	VkImageSubresourceRange subresource_range = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0,
		.levelCount = 1,
		.baseArrayLayer = 0,
		.layerCount = 1
	};

	IMAGE_BARRIER(cmd_buf,
		.image = src_image,
		.subresourceRange = subresource_range,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
	);

	IMAGE_BARRIER(cmd_buf,
		.image = capture.image,
		.subresourceRange = subresource_range,
		.srcAccessMask = VK_ACCESS_HOST_READ_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
	);

	VkImageCopy image_copy_region = {
		.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.extent = { extent.width, extent.height, 1 }
	};

	vkCmdCopyImage(cmd_buf,
		src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		capture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &image_copy_region);

	IMAGE_BARRIER(cmd_buf,
		.image = src_image,
		.subresourceRange = subresource_range,
		.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL
	);

	IMAGE_BARRIER(cmd_buf,
		.image = capture.image,
		.subresourceRange = subresource_range,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL
	);

	capture.recorded_extent = extent;
	capture.frame = qvk.frame_counter;

	if (vkpt_refdef.fd) {
		VectorCopy(vkpt_refdef.fd->vieworg, capture.vieworg);
		VectorCopy(vkpt_refdef.fd->viewangles, capture.viewangles);
	} else {
		VectorClear(capture.vieworg);
		VectorClear(capture.viewangles);
	}

	capture.recorded = true;
}

void vkpt_capture_writeout(void)
{
	if (!capture.recorded)
		return;

	capture.recorded = false;
	capture.requested = false;

	// The copy above is in a command buffer that has just been submitted; the
	// mapped memory is not valid until the GPU is done with it.
	_VK(vkQueueWaitIdle(qvk.queue_graphics));

	VkImageSubresource subresource = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.arrayLayer = 0,
		.mipLevel = 0
	};

	VkSubresourceLayout subresource_layout;
	vkGetImageSubresourceLayout(qvk.device, capture.image, &subresource, &subresource_layout);

	void *data;
	_VK(vkMapMemory(qvk.device, capture.memory, 0, capture.memory_size, 0, &data));

	const byte *pixels = (const byte *)data + subresource_layout.offset;

	uint32_t tile = (uint32_t)Q_clip(cvar_capture_tile->integer, 16, CAPTURE_MAX_TILE);
	int wanted = Q_clip(cvar_capture_tiles_per_frame->integer, 1, CAPTURE_MAX_TILES);
	float min_stddev = max(0.f, cvar_capture_min_stddev->value);

	const char *tag = cvar_capture_tag->string;
	if (!*tag)
		tag = "capture";

	// Tiles are taken from a grid of tile-sized cells rather than from random
	// positions. Two crops that overlap are one training sample's worth of
	// information stored twice, and rejection-sampling random positions until
	// they happen not to overlap finds only a fraction of the tiles a frame can
	// actually yield: a 3840x2160 frame holds 7x4 disjoint 512 tiles, and
	// picking those by chance takes far more attempts than it is worth.
	//
	// The grid is offset by a random amount inside the leftover margin, so
	// repeated captures of the same view do not keep cropping on exactly the
	// same pixel boundaries.
	uint32_t cells_x = capture.recorded_extent.width / tile;
	uint32_t cells_y = capture.recorded_extent.height / tile;
	uint32_t cells = cells_x * cells_y;

	uint32_t jitter_x = Q_rand() % (capture.recorded_extent.width - cells_x * tile + 1);
	uint32_t jitter_y = Q_rand() % (capture.recorded_extent.height - cells_y * tile + 1);

	// Visit the cells in a random order and keep the first 'wanted' that hold
	// something worth training on, so a frame whose interesting content is all
	// in one corner is not systematically sampled from the same corner.
	uint32_t *order = Z_Malloc(cells * sizeof(*order));
	for (uint32_t i = 0; i < cells; i++)
		order[i] = i;
	for (uint32_t i = cells; i > 1; i--) {
		uint32_t j = Q_rand() % i;
		uint32_t swap = order[i - 1];
		order[i - 1] = order[j];
		order[j] = swap;
	}

	int written = 0;

	for (uint32_t i = 0; i < cells && written < wanted; i++) {
		uint32_t x = jitter_x + (order[i] % cells_x) * tile;
		uint32_t y = jitter_y + (order[i] / cells_x) * tile;

		if (tile_stddev(pixels, subresource_layout.rowPitch, x, y, tile) < min_stddev)
			continue;

		if (write_tile(pixels, subresource_layout.rowPitch, x, y, tile, tag))
			written++;
	}

	Z_Free(order);
	vkUnmapMemory(qvk.device, capture.memory);

	capture.tiles_written += written;

	// Reported every time rather than only on failure: an unattended run over
	// forty maps is otherwise completely silent about whether it is producing
	// anything.
	Com_Printf("capture: %s frame %" PRIu64 ": wrote %d of %d tiles (%u total)\n",
		tag, capture.frame, written, wanted, capture.tiles_written);
}

unsigned vkpt_capture_tiles_written(void)
{
	return capture.tiles_written;
}
