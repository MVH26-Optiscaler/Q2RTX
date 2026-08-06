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
	AI upscaler ("upscaler") implementation overview
	=================================================

	Runs a QuickSRNet super-resolution model (w8a8, ONNX) through ONNX Runtime's
	Qualcomm QNN Execution Provider, so inference dispatches to the Hexagon NPU
	on Windows-ARM64 (Snapdragon) devices.

	Per frame the tone-mapped TAA output is blitted into an R8G8B8A8_SRGB image
	and copied straight into the model's input tensor; the CPU runs one inference
	for the whole frame on the NPU; the result is copied straight back into an
	image the final blit samples. The GPU->NPU->GPU round trip is a hard pipeline
	stall, which is why it is pipelined a frame deep rather than waited on.

	No pack/unpack shaders
	----------------------
	The models take and return uint8 NHWC RGBA -- 1 x H x W x 4, which is
	byte-for-byte a tightly packed 8-bit RGBA image. So the transfer in each
	direction is one vkCmdCopyImageToBuffer / vkCmdCopyBufferToImage with no
	shader in between, and the float->unorm8 conversion, the [0,1] clamp and the
	linear->sRGB encode all fall out of the blit that feeds it.

	Colour space
	------------
	Tone mapping leaves TAA_OUTPUT linear: it clamps to [0,1] and dithers for an
	8-bit sRGB quantization, but the sRGB encode itself is the hardware's, applied
	when the final blit writes the (always sRGB) swapchain view. Quantizing that
	linear signal to 8 bits for the tensor would hand the model a transfer
	function it was never trained on and spend half the code words on highlights,
	so the round-trip images are sRGB instead and the hardware encodes on the way
	in and decodes on the way out. See create_ldr_image().

	Models as trained are NCHW; scripts/onnx-nhwc-io.py wraps their graph I/O in a
	pair of Transpose nodes to move the declared layout, which ONNX Runtime's
	transpose optimizer then folds away against the NHWC graph the QNN EP builds
	anyway. Feeding an NCHW model interleaved pixels would render colour bands
	rather than an image, so load_session() refuses anything whose input is not
	[1, H, W, 4].

	Resolution
	----------
	The model's scale factor is a property of the model, not a resolution policy.
	viewsize/DRS choose the render extent exactly as they do for every other path;
	the model multiplies it by its fixed factor, and whatever that overshoots the
	display by, the final blit removes on the way out by filtering.

	So the downsample ratio is viewsize * scale / 100. With a 4x model, viewsize 25
	lands on the display exactly and the blit is a 1:1 copy -- the cheap
	upscale-for-performance case. Above that the extra resolution is real
	supersampling, and at viewsize 100 the path tracer runs at native resolution
	and the frame is 4x-downsampled: high quality, and far too slow for gameplay.

	Shapes and the session
	----------------------
	The models declare H and W as free dimensions, but the HTP wants a static
	graph, so the render extent is pinned into the session with
	AddFreeDimensionOverrideByName() and the session becomes specific to it.
	Changing viewsize or resolution therefore costs a session rebuild, which takes
	seconds because the QNN EP finalizes the HTP graph. Dynamic render scaling
	moves the extent every frame and cannot be served at any useful rate, so
	ensure_session() refuses to run alongside it rather than stalling repeatedly.

	Q2RTX cvars
	-----------
	* flt_upscaler_enable - which model to run: 0 = disabled, 1 = QuickSRNet
	  Large 2x, 2 = QuickSRNet Large 4x (see upscaler_models[]). Changing it drops
	  the ONNX Runtime session; the next frame rebuilds it, which takes a few
	  seconds. Normally driven by the flt_upscaling menu cvar.
	* flt_upscaler_verbose - raise ONNX Runtime logging to verbose, which is
	  where per-node execution-provider assignment is reported.

	Console commands
	----------------
	* upscaler_npu_test [seconds] - run repeated inference for the given
	  duration (default 3s) so NPU dispatch/utilization can be confirmed via
	  Windows Task Manager > Performance > NPU while it runs, and via the
	  verbose ONNX Runtime log lines this prints to the console.
	* upscaler_dump - dump the pre- and post-upscale tensor of the next rendered
	  frame as PNGs under <gamedir>/screenshots/upscaler/, for visually
	  inspecting what goes into and comes out of the model.
*/

#include "shared/shared.h"
#include "common/common.h"
#include "vkpt.h"

#include <string.h>
#include <stdlib.h>

cvar_t *cvar_flt_upscaler_enable = NULL;
cvar_t *cvar_flt_upscaler_verbose = NULL;

extern cvar_t *cvar_flt_fsr_enable; // owned by fsr.c, initialized just before us
extern cvar_t *cvar_drs_enable;     // owned by main.c; DRS and this upscaler are exclusive

cvar_t *cvar_flt_upscaling = NULL;

// The NPU upscaler models, indexed by flt_upscaler_enable - 1. Both ship in the
// repo under baseq2/models, both are QuickSRNet Large fine-tuned on Quake II RTX
// frames, quantized to w8a8 and re-exported with uint8 NHWC RGBA I/O and free
// H/W dimensions.
//
// The scale factor is declared here rather than discovered, because the frame
// graph has to settle whether the upscaler is in it -- upscaler_active_this_frame()
// in main.c keys off vkpt_upscaler_get_scale() -- before there is a session to
// ask. It is not taken on trust: load_session() runs one inference and refuses
// the model if what comes back is not this factor.
//
// Order matters: it defines both the flt_upscaler_enable values and the
// flt_upscaling values below, so append rather than insert.
static const struct {
	const char *name;
	const char *path;
	uint32_t    scale;
} upscaler_models[] = {
	{ "QuickSRNet Large 2x", "models/quicksrnetlarge-2x-rgba-w8a8.onnx", 2 },
	{ "QuickSRNet Large 4x", "models/quicksrnetlarge-4x-rgba-w8a8.onnx", 4 },
};

// Menu-facing selector for the mutually exclusive upscalers. The per-backend
// cvars stay authoritative so existing configs, scripts and console use keep
// working; this just keeps them from being enabled at the same time.
enum {
	UPSCALING_MODE_NONE     = 0,
	UPSCALING_MODE_FSR      = 1,
	UPSCALING_MODE_AI_FIRST = 2, // 2 .. 2 + <number of models> - 1
};

// Drops whatever session is loaded when flt_upscaler_enable names a different
// model. No-op until the ONNX Runtime side is up.
static void upscaler_reload_model(void);

static void upscaling_mode_changed(cvar_t *self)
{
	// flt_upscaler_enable is a 1-based model index rather than a boolean, so
	// everything that only asks "is the AI upscaler on?" still just tests it
	// against zero.
	int model = self->integer - (UPSCALING_MODE_AI_FIRST - 1);
	if (model < 1 || model > (int)LENGTH(upscaler_models))
		model = 0;

	Cvar_SetInteger(cvar_flt_fsr_enable, self->integer == UPSCALING_MODE_FSR, FROM_CODE);
	Cvar_SetInteger(cvar_flt_upscaler_enable, model, FROM_CODE);

	// Cvar_SetInteger(FROM_CODE) deliberately does not run change callbacks
	// (change_string_value() in common/cvar.c), so the swap has to be kicked
	// off from here rather than from flt_upscaler_enable's own callback.
	upscaler_reload_model();
}

void vkpt_upscaler_init_cvars(void)
{
	cvar_flt_upscaler_enable = Cvar_Get("flt_upscaler_enable", "0", CVAR_ARCHIVE);
	// Dumps ONNX Runtime's per-node execution-provider assignment at startup,
	// which is how you confirm the model is really running on the NPU -- and, for
	// this branch specifically, that no Transpose was left behind on the CPU.
	cvar_flt_upscaler_verbose = Cvar_Get("flt_upscaler_verbose", "0", 0);

	// upscaling_mode_changed() below overwrites flt_upscaler_enable from the menu
	// cvar, so latch the archived value first.
	int archived_model = cvar_flt_upscaler_enable->integer;

	// Seeded from whatever the backend cvars already say, so a config that set
	// flt_fsr_enable or flt_upscaler_enable directly shows up correctly in the
	// menu, on the right model.
	char initial[16];
	if (archived_model > 0 && archived_model <= (int)LENGTH(upscaler_models))
	{
		Q_snprintf(initial, sizeof(initial), "%d",
			UPSCALING_MODE_AI_FIRST - 1 + archived_model);
	} else {
		Q_strlcpy(initial, cvar_flt_fsr_enable && cvar_flt_fsr_enable->integer ? "1" : "0", sizeof(initial));
	}

	cvar_flt_upscaling = Cvar_Get("flt_upscaling", initial, CVAR_ARCHIVE);
	cvar_flt_upscaling->changed = upscaling_mode_changed;
	upscaling_mode_changed(cvar_flt_upscaling);
}

#ifdef USE_ORT_QNN_UPSCALER

#include <windows.h>
#include "onnxruntime_c_api.h"
#include "vk_util.h"
#include "stb_image_write.h"

#define UPSCALER_INPUT_NAME "image"
#define UPSCALER_OUTPUT_NAME "upscaled_image"
#define UPSCALER_NUM_DIMS 4
#define UPSCALER_CHANNELS 4
#define UPSCALER_DEFAULT_TEST_SECONDS 3

// Sanity bound on the render extent, so a bogus resolution cannot ask for an
// absurd host-visible allocation before anything else notices.
#define UPSCALER_MAX_EDGE 8192

// Minimum gap between session rebuilds. The rebuild finalizes the HTP graph and
// takes seconds, so a burst of extent changes -- dragging the window, holding
// down the viewsize slider -- must not queue one up per frame.
#define UPSCALER_REBUILD_INTERVAL_MS 1000

// One 8-bit RGBA image the transfer commands work against, sRGB where the device
// allows it -- see create_ldr_image(). These are not VKPT_IMG_* screen images:
// the output one is the render extent times the model scale, which at viewsize
// 100 with a 4x model is four times the extent get_screen_image_extent()
// allocates.
typedef struct {
	VkImage        image;
	VkImageView    view;
	VkDeviceMemory memory;
	VkExtent2D     extent;
} upscaler_image_t;

// One frame's worth of the GPU <-> NPU round trip. The round trip is pipelined
// rather than stalled on: frame N downloads into slot N%2 and moves on, frame
// N+1 runs the inference for slot N%2 and uploads it. So the displayed upscale
// is one frame old, and the CPU waits on download_fence -- which the previous
// frame's submit already signalled -- instead of draining the queue.
//
// Two slots suffice:
//  - buf_input is GPU-written in frame N and CPU-read in frame N+1 under
//    download_fence; the next GPU write to it is frame N+2's download, recorded
//    after that read returned on the same thread.
//  - buf_output is CPU-written in frame N and read by the upload submitted in
//    frame N; the next CPU write is frame N+2, and R_BeginFrame_RTX's wait on
//    fences_frame_sync[current_frame_index] has by then guaranteed all of
//    frame N completed. That is also why reallocating a slot on an extent change
//    is safe: the last GPU reference to it was frame N-2's.
typedef struct {
	BufferResource_t buf_input;
	BufferResource_t buf_output;
	void            *input_mapped;
	void            *output_mapped;
	upscaler_image_t img_in;   // render extent, blit target then copy source
	upscaler_image_t img_out;  // render extent * scale, copy target then sampled
	// Captured when the download is recorded and replayed at upload time:
	// viewsize can change the extent between the two frames, and the upload has
	// to describe the extent it was downloaded at, not the current one.
	VkExtent2D       extent_in;
	VkExtent2D       extent_out;
	// Signalled by whichever submit carries this slot's download. The only thing
	// tracking "the download finished" -- vkpt_submit_command_buffer_simple()
	// passes no fence.
	VkFence          download_fence;
	bool             fence_pending;  // handed to a submit, not waited on since
	bool             downloaded;     // download submitted, inference not run yet
	bool             tensor_valid;   // inference produced a usable result
	bool             dump;           // dump_requested, captured at download time
} upscaler_slot_t;

struct
{
	const OrtApi  *api;
	OrtEnv        *env;
	OrtSession    *session;
	OrtMemoryInfo *cpu_memory_info;
	bool           initialized;   // env/allocator are up; safe to load models
	int            selected_model; // 1-based index into upscaler_models, 0 = none
	bool           session_ready;  // session matches session_extent and works
	bool           session_failed; // load failed; do not retry until the selection changes
	unsigned       last_rebuild_ms;

	VkExtent2D     session_extent; // extent the free dimensions were pinned to
	int64_t        input_dims[UPSCALER_NUM_DIMS];
	int64_t        output_dims[UPSCALER_NUM_DIMS];
	size_t         input_byte_size;  // uint8 tensor, 1 byte/element
	size_t         output_byte_size;

	// Render integration.
	bool             ready; // fences are up; the pass may be recorded
	upscaler_slot_t  slots[MAX_FRAMES_IN_FLIGHT];

	// Format of img_in/img_out, chosen once by pick_ldr_format(). Normally
	// R8G8B8A8_SRGB so the hardware does the transfer-function conversion; falls
	// back to R8G8B8A8_UNORM on a device that cannot blit into sRGB.
	VkFormat         ldr_format;

	// One-shot flag set by the upscaler_dump console command, captured into the
	// slot by spatial_download() so the pre/post pair covers the same frame.
	bool             dump_requested;

	// Rolling cost of the NPU round trip. Accumulated over many frames because
	// Sys_Milliseconds() truncates to whole milliseconds: a single sample carries
	// +-1 ms, but the tick phase is uncorrelated with the frame loop so the
	// average over UPSCALER_TIMING_INTERVAL frames converges.
	//
	// Only ever covers one model at one extent: reset_timing() clears these
	// whenever the session is torn down, which is the only way either can change.
	// Without that an interval straddling a switch would average two
	// configurations and report the result under whichever one happened to be
	// current at the end.
	//
	// stall_ms_accum is the wait on the previous frame's download_fence. That
	// fence is signalled by that frame's whole post submit, so it measures how far
	// the GPU is behind -- not the cost of the download transfer itself.
	unsigned         inference_ms_accum;
	unsigned         stall_ms_accum;
	unsigned         inference_frames;
} upscaler;

#define UPSCALER_TIMING_INTERVAL 100 // frames between timing reports

static void reset_timing(void)
{
	upscaler.inference_ms_accum = 0;
	upscaler.stall_ms_accum = 0;
	upscaler.inference_frames = 0;
}

static inline bool upscaler_extents_equal(VkExtent2D a, VkExtent2D b)
{
	return a.width == b.width && a.height == b.height;
}

// The model the selector names, or NULL when it names none.
static inline int selected_model_index(void)
{
	int index = cvar_flt_upscaler_enable->integer;
	return (index >= 1 && index <= (int)LENGTH(upscaler_models)) ? index : 0;
}

// Whether the pass may run at all this frame. Deliberately does not depend on
// the session: the session is built lazily against the render extent, and the
// render extent is derived from the scale factor this gates, so making it wait
// for a session would be circular. A frame that finds no usable session falls
// back to the filtered blit in vkpt_upscaler_final_blit().
static bool spatial_is_active(void)
{
	return upscaler.initialized && upscaler.ready
		&& upscaler.selected_model != 0 && !upscaler.session_failed;
}

// The pipelined round trip indexes its staging by qvk.current_frame_index: the
// frame that downloads owns `cur`, and the same frame runs the inference for --
// and uploads -- what the previous one left in `prev`.
static upscaler_slot_t *spatial_slot_cur(void)
{
	return &upscaler.slots[qvk.current_frame_index];
}

static upscaler_slot_t *spatial_slot_prev(void)
{
	return &upscaler.slots[(qvk.current_frame_index + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT];
}

// Drops any in-flight tensor state. Both flags gate every use of the staging
// buffers, so clearing them is enough to make a slot inert -- needed whenever
// the frame that would have consumed a downloaded tensor does not run (menu
// mode, FSR taking over, accumulation rendering) or the buffers behind it are
// about to be rebuilt at a different extent.
void vkpt_upscaler_discard(void)
{
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		upscaler.slots[i].downloaded = false;
		upscaler.slots[i].tensor_valid = false;
	}
}

static void ORT_API_CALL upscaler_ort_log(void *param, OrtLoggingLevel severity, const char *category,
	const char *logid, const char *code_location, const char *message)
{
	if (severity >= ORT_LOGGING_LEVEL_ERROR)
		Com_EPrintf("upscaler: [ORT] %s: %s\n", code_location, message);
	else
		Com_Printf("upscaler: [ORT] %s: %s\n", code_location, message);
}

static bool ort_ok(OrtStatus *status, const char *what)
{
	if (!status)
		return true;

	Com_EPrintf("upscaler: %s failed: %s\n", what, upscaler.api->GetErrorMessage(status));
	upscaler.api->ReleaseStatus(status);
	return false;
}

// Queries the shape and element type of one input/output tensor of `session`.
// The models are quantized at the graph boundary, so elem_type is always UINT8.
// Byte sizes are not derived here: with free dimensions in play the reported
// shape can still hold -1s, so the caller computes them from the extent it
// pinned instead.
static bool query_tensor_shape(OrtSession *session, bool is_input, size_t index, const char *expected_name,
	int64_t *dims_out)
{
	OrtAllocator *allocator;
	if (!ort_ok(upscaler.api->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions"))
		return false;

	char *name = NULL;
	OrtTypeInfo *type_info = NULL;
	bool ok = false;

	OrtStatus *status = is_input
		? upscaler.api->SessionGetInputName(session, index, allocator, &name)
		: upscaler.api->SessionGetOutputName(session, index, allocator, &name);
	if (!ort_ok(status, "SessionGetInput/OutputName"))
		return false;

	if (strcmp(name, expected_name) != 0) {
		Com_WPrintf("upscaler: model %s %zu is named '%s', expected '%s'\n",
			is_input ? "input" : "output", index, name, expected_name);
	}

	status = is_input
		? upscaler.api->SessionGetInputTypeInfo(session, index, &type_info)
		: upscaler.api->SessionGetOutputTypeInfo(session, index, &type_info);
	if (!ort_ok(status, "SessionGetInput/OutputTypeInfo"))
		goto done;

	{
		const OrtTensorTypeAndShapeInfo *tensor_info;
		if (!ort_ok(upscaler.api->CastTypeInfoToTensorInfo(type_info, &tensor_info), "CastTypeInfoToTensorInfo"))
			goto done;

		size_t num_dims;
		if (!ort_ok(upscaler.api->GetDimensionsCount(tensor_info, &num_dims), "GetDimensionsCount"))
			goto done;

		if (num_dims != UPSCALER_NUM_DIMS) {
			Com_EPrintf("upscaler: expected a %d-D tensor, model %s %zu has %zu dims\n",
				UPSCALER_NUM_DIMS, is_input ? "input" : "output", index, num_dims);
			goto done;
		}

		if (!ort_ok(upscaler.api->GetDimensions(tensor_info, dims_out, num_dims), "GetDimensions"))
			goto done;

		ONNXTensorElementDataType elem_type;
		if (!ort_ok(upscaler.api->GetTensorElementType(tensor_info, &elem_type), "GetTensorElementType"))
			goto done;

		if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
			Com_EPrintf("upscaler: expected uint8 tensors, model %s %zu has element type %d\n",
				is_input ? "input" : "output", index, (int)elem_type);
			goto done;
		}

		Com_Printf("upscaler: model %s '%s' shape [%lld,%lld,%lld,%lld]\n",
			is_input ? "input" : "output", name,
			(long long)dims_out[0], (long long)dims_out[1], (long long)dims_out[2], (long long)dims_out[3]);

		ok = true;
	}

done:
	if (type_info)
		upscaler.api->ReleaseTypeInfo(type_info);
	if (name)
		upscaler.api->AllocatorFree(allocator, name);
	return ok;
}

// Phase 1 verification harness: runs repeated inference on a flat mid-gray
// test image for the requested duration, so NPU dispatch and utilization can
// be observed live (Task Manager > Performance > NPU) and the per-inference
// timing can be sanity-checked against Qualcomm's published benchmarks.
static void Upscaler_NpuTest_f(void)
{
	if (!upscaler.session_ready) {
		Com_Printf("upscaler: no NPU upscaler session loaded, nothing to test "
			"(the session is built from the render extent, so render a frame with "
			"flt_upscaling set to an AI model first)\n");
		return;
	}

	int duration_ms = (Cmd_Argc() > 1 ? atoi(Cmd_Argv(1)) : UPSCALER_DEFAULT_TEST_SECONDS) * 1000;
	if (duration_ms <= 0)
		duration_ms = UPSCALER_DEFAULT_TEST_SECONDS * 1000;

	void *input_data = Z_Mallocz(upscaler.input_byte_size);
	void *output_data = Z_Mallocz(upscaler.output_byte_size);
	memset(input_data, 128, upscaler.input_byte_size);

	OrtValue *input_value = NULL, *output_value = NULL;
	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, input_data,
			upscaler.input_byte_size, upscaler.input_dims, UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &input_value), "CreateTensorWithDataAsOrtValue(input)"))
		goto done;

	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, output_data,
			upscaler.output_byte_size, upscaler.output_dims, UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &output_value), "CreateTensorWithDataAsOrtValue(output)"))
		goto done;

	{
		const char *input_names[]  = { UPSCALER_INPUT_NAME };
		const char *output_names[] = { UPSCALER_OUTPUT_NAME };
		const OrtValue *inputs[]   = { input_value };
		OrtValue *outputs[]        = { output_value };

		Com_Printf("upscaler: running NPU test inference at %ux%u for %d ms "
			"(watch Task Manager > Performance > NPU)\n",
			upscaler.session_extent.width, upscaler.session_extent.height, duration_ms);

		unsigned start = Sys_Milliseconds();
		unsigned elapsed = 0;
		int iterations = 0;
		while ((int)elapsed < duration_ms) {
			OrtStatus *status = upscaler.api->Run(upscaler.session, NULL,
				input_names, inputs, 1, output_names, 1, outputs);
			if (status) {
				Com_EPrintf("upscaler: NPU test inference failed: %s\n", upscaler.api->GetErrorMessage(status));
				upscaler.api->ReleaseStatus(status);
				break;
			}
			iterations++;
			elapsed = Sys_Milliseconds() - start;
		}

		if (iterations > 0) {
			Com_Printf("upscaler: ran %d inferences in %u ms (avg %.3f ms/inference)\n",
				iterations, elapsed, (double)elapsed / iterations);
		}
	}

done:
	if (input_value)
		upscaler.api->ReleaseValue(input_value);
	if (output_value)
		upscaler.api->ReleaseValue(output_value);
	Z_Free(input_data);
	Z_Free(output_data);
}

// Writes one RGBA tensor out as a PNG, for visually inspecting what actually
// goes into/comes out of the model. Triggered by the "upscaler_dump" console
// command via spatial_run_inference(). The tensor is interleaved RGBA8 already,
// so this is a straight write -- a dump that comes out as horizontal colour
// bands means an NCHW model slipped past load_session().
//
// The bytes are sRGB-encoded (see create_ldr_image()), which is what PNG means
// by 8-bit colour, so the dumps are directly comparable to a screenshot. A dump
// that looks uniformly too dark means the round trip fell back to linear.
static void upscaler_dump_tensor(const char *stage, const uint8_t *rgba, VkExtent2D extent)
{
	char path[MAX_OSPATH];
	if (Q_snprintf(path, sizeof(path), "%s/screenshots/upscaler/upscaler_%s_%" PRIu64 ".png",
			fs_gamedir, stage, qvk.frame_counter) >= sizeof(path))
	{
		Com_EPrintf("upscaler: dump path too long\n");
		return;
	}

	if (FS_CreatePath(path) < 0) {
		Com_EPrintf("upscaler: failed to create directory for '%s'\n", path);
		return;
	}

	if (!stbi_write_png(path, extent.width, extent.height, 4, rgba, extent.width * 4))
		Com_EPrintf("upscaler: failed to write '%s'\n", path);
	else
		Com_Printf("upscaler: wrote %s (%ux%u)\n", path, extent.width, extent.height);
}

static void Upscaler_Dump_f(void)
{
	upscaler.dump_requested = true;
	Com_Printf("upscaler: will dump pre/post-upscale tensors for the next frame\n");
}

static const cmdreg_t upscaler_cmds[] = {
	{ "upscaler_npu_test", &Upscaler_NpuTest_f, NULL },
	{ "upscaler_dump", &Upscaler_Dump_f, NULL },
	{ NULL, NULL, NULL }
};

static void destroy_slots(void); // defined with the render integration below

// Session options + the QNN HTP execution provider + CreateSession, given a
// model file relative to the game dir and the extent to pin its free dimensions
// to.
static bool create_qnn_session(const char *model_file, const char *display_name, VkExtent2D extent,
	OrtSession **out_session)
{
	OrtSessionOptions *session_options = NULL;
	if (!ort_ok(upscaler.api->CreateSessionOptions(&session_options), "CreateSessionOptions"))
		return false;

	{
		const char *provider_keys[]   = { "backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode" };
		const char *provider_values[] = { "QnnHtp.dll", "high_performance", "3" };
		OrtStatus *ep_status = upscaler.api->SessionOptionsAppendExecutionProvider(session_options, "QNN",
			provider_keys, provider_values, LENGTH(provider_keys));
		if (!ort_ok(ep_status, "SessionOptionsAppendExecutionProvider(QNN)")) {
			Com_Printf("upscaler: QNN execution provider unavailable, NPU upscaler disabled\n");
			upscaler.api->ReleaseSessionOptions(session_options);
			return false;
		}
	}

	// The HTP wants a static graph. Pinning the model's free dimensions before
	// partitioning is what lets the QNN EP take the whole thing instead of
	// leaving it on the CPU -- and it is why the session is specific to one
	// render extent.
	if (!ort_ok(upscaler.api->AddFreeDimensionOverrideByName(session_options, "height", extent.height),
			"AddFreeDimensionOverrideByName(height)") ||
		!ort_ok(upscaler.api->AddFreeDimensionOverrideByName(session_options, "width", extent.width),
			"AddFreeDimensionOverrideByName(width)"))
	{
		upscaler.api->ReleaseSessionOptions(session_options);
		return false;
	}

	char model_path[MAX_OSPATH];
	if (Q_concat(model_path, sizeof(model_path), fs_gamedir, PATH_SEP_STRING, model_file) >= sizeof(model_path)) {
		Com_EPrintf("upscaler: model path too long\n");
		upscaler.api->ReleaseSessionOptions(session_options);
		return false;
	}

	Com_Printf("upscaler: loading %s (%s) at %ux%u, this takes a moment...\n",
		model_file, display_name, extent.width, extent.height);

	WCHAR wmodel_path[MAX_OSPATH];
	MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wmodel_path, MAX_OSPATH);

	OrtStatus *session_status = upscaler.api->CreateSession(upscaler.env, wmodel_path, session_options, out_session);
	upscaler.api->ReleaseSessionOptions(session_options);

	if (!ort_ok(session_status, "CreateSession")) {
		Com_Printf("upscaler: could not load %s, %s unavailable "
			"(the models ship in baseq2/models; restore with 'git checkout -- baseq2/models')\n",
			model_path, display_name);
		return false;
	}
	return true;
}

// Runs one inference with an ONNX Runtime-allocated output, so the output shape
// -- whose dimensions are free and therefore report as -1 on the session -- can
// be read off the result. Doubles as the check that the model really does what
// upscaler_models[] claims, and warms the HTP graph.
static bool probe_output_shape(uint32_t expected_scale, VkExtent2D extent)
{
	uint8_t *input_data = Z_Mallocz(upscaler.input_byte_size);

	OrtValue *input_value = NULL;
	OrtValue *output_value = NULL;
	bool ok = false;

	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, input_data,
			upscaler.input_byte_size, upscaler.input_dims, UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &input_value), "CreateTensorWithDataAsOrtValue(probe input)"))
		goto done;

	{
		const char *input_names[]  = { UPSCALER_INPUT_NAME };
		const char *output_names[] = { UPSCALER_OUTPUT_NAME };
		const OrtValue *inputs[]   = { input_value };

		if (!ort_ok(upscaler.api->Run(upscaler.session, NULL, input_names, inputs, 1,
				output_names, 1, &output_value), "probe inference"))
			goto done;
	}

	{
		OrtTensorTypeAndShapeInfo *info = NULL;
		if (!ort_ok(upscaler.api->GetTensorTypeAndShape(output_value, &info), "GetTensorTypeAndShape"))
			goto done;

		OrtStatus *status = upscaler.api->GetDimensions(info, upscaler.output_dims, UPSCALER_NUM_DIMS);
		upscaler.api->ReleaseTensorTypeAndShapeInfo(info);
		if (!ort_ok(status, "GetDimensions(probe output)"))
			goto done;
	}

	if (upscaler.output_dims[0] != 1 || upscaler.output_dims[3] != UPSCALER_CHANNELS ||
		upscaler.output_dims[1] != (int64_t)extent.height * expected_scale ||
		upscaler.output_dims[2] != (int64_t)extent.width * expected_scale)
	{
		Com_EPrintf("upscaler: model returned [%lld,%lld,%lld,%lld] for a %ux%u input, "
			"expected [1,%u,%u,%d] for a %ux upscale; not loading\n",
			(long long)upscaler.output_dims[0], (long long)upscaler.output_dims[1],
			(long long)upscaler.output_dims[2], (long long)upscaler.output_dims[3],
			extent.width, extent.height,
			extent.height * expected_scale, extent.width * expected_scale, UPSCALER_CHANNELS,
			expected_scale);
		goto done;
	}

	upscaler.output_byte_size = (size_t)extent.width * expected_scale
		* (size_t)extent.height * expected_scale * UPSCALER_CHANNELS;
	ok = true;

done:
	if (input_value)
		upscaler.api->ReleaseValue(input_value);
	if (output_value)
		upscaler.api->ReleaseValue(output_value);
	Z_Free(input_data);
	return ok;
}

// Releases the session. The env and the CPU allocator outlive it, so switching
// models or extents does not rebuild them.
static void unload_session(void)
{
	if (upscaler.session) {
		upscaler.api->ReleaseSession(upscaler.session);
		upscaler.session = NULL;
	}

	// The staging buffers and images are sized from the session's extent, so they
	// cannot outlive it. The transfers referencing them may still be in flight,
	// hence the wait.
	if (qvk.device)
		vkDeviceWaitIdle(qvk.device);
	destroy_slots();

	// Whatever has been accumulated describes the session being torn down. This
	// is the choke point for every change that matters -- a different model, or a
	// different render extent, both of which can only take effect through a
	// rebuild -- so clearing here is what keeps a reported average describing one
	// configuration rather than a blend of two.
	reset_timing();

	upscaler.session_ready = false;
	upscaler.session_extent = (VkExtent2D){ 0, 0 };
	upscaler.input_byte_size = 0;
	upscaler.output_byte_size = 0;
}

// Builds a session for the selected model at `extent`. Expensive -- the QNN EP
// finalizes the HTP graph, which takes seconds -- so this deliberately stalls
// rather than trying to hide the switch.
static bool load_session(VkExtent2D extent)
{
	unload_session();

	int index = upscaler.selected_model;
	if (index < 1 || index > (int)LENGTH(upscaler_models))
		return false;

	const char *model_file   = upscaler_models[index - 1].path;
	const char *display_name = upscaler_models[index - 1].name;
	uint32_t    scale        = upscaler_models[index - 1].scale;

	if (extent.width == 0 || extent.height == 0 ||
		extent.width > UPSCALER_MAX_EDGE || extent.height > UPSCALER_MAX_EDGE ||
		extent.width * scale > UPSCALER_MAX_EDGE || extent.height * scale > UPSCALER_MAX_EDGE)
	{
		Com_EPrintf("upscaler: %ux%u at %ux is outside the supported range (max edge %d)\n",
			extent.width, extent.height, scale, UPSCALER_MAX_EDGE);
		return false;
	}

	if (!create_qnn_session(model_file, display_name, extent, &upscaler.session))
		return false;

	if (!query_tensor_shape(upscaler.session, true, 0, UPSCALER_INPUT_NAME, upscaler.input_dims))
	{
		unload_session();
		return false;
	}

	// The whole copy-based transfer rests on the tensor being interleaved RGBA,
	// i.e. NHWC. An NCHW model would load and run and produce colour bands, so
	// the channel count has to be in the last dimension, not the second.
	if (upscaler.input_dims[0] != 1 || upscaler.input_dims[3] != UPSCALER_CHANNELS) {
		Com_EPrintf("upscaler: %s input is [%lld,%lld,%lld,%lld], expected [1,H,W,%d] NHWC RGBA. "
			"Convert it with scripts/onnx-nhwc-io.py; not loading\n", model_file,
			(long long)upscaler.input_dims[0], (long long)upscaler.input_dims[1],
			(long long)upscaler.input_dims[2], (long long)upscaler.input_dims[3],
			UPSCALER_CHANNELS);
		unload_session();
		return false;
	}

	// Whether the override made it into the reported shape or left it free, the
	// tensor we hand in is the one that decides. Only reject an explicitly
	// different concrete size.
	if ((upscaler.input_dims[1] > 0 && upscaler.input_dims[1] != (int64_t)extent.height) ||
		(upscaler.input_dims[2] > 0 && upscaler.input_dims[2] != (int64_t)extent.width))
	{
		Com_EPrintf("upscaler: %s did not take the %ux%u free-dimension override "
			"(input is %lldx%lld); not loading\n", model_file, extent.width, extent.height,
			(long long)upscaler.input_dims[2], (long long)upscaler.input_dims[1]);
		unload_session();
		return false;
	}

	upscaler.input_dims[1] = extent.height;
	upscaler.input_dims[2] = extent.width;
	upscaler.input_byte_size = (size_t)extent.width * extent.height * UPSCALER_CHANNELS;

	if (!probe_output_shape(scale, extent)) {
		unload_session();
		return false;
	}

	upscaler.session_extent = extent;
	upscaler.session_ready = true;

	Com_Printf("upscaler: %s loaded, %ux%u -> %ux%u (%.1f MB staging per frame in flight); "
		"run 'upscaler_npu_test' to verify NPU dispatch\n",
		display_name, extent.width, extent.height, extent.width * scale, extent.height * scale,
		(double)(upscaler.input_byte_size + upscaler.output_byte_size) / (1024.0 * 1024.0));

	return true;
}

// Brings the loaded session in line with the selector. Reached from the menu,
// via upscaling_mode_changed(), and from setting either backend cvar straight
// from the console. The new session is not built here -- it needs the render
// extent, which the next frame supplies.
static void upscaler_reload_model(void)
{
	if (!upscaler.initialized)
		return;

	int index = selected_model_index();
	if (index == upscaler.selected_model)
		return;

	upscaler.selected_model = index;
	upscaler.session_failed = false;
	upscaler.last_rebuild_ms = 0;
	unload_session();
	vkpt_upscaler_discard();
}

static void upscaler_model_changed(cvar_t *self)
{
	upscaler_reload_model();
}

// The transfer function the model round trip runs in. R8G8B8A8_SRGB gets the
// encode and the decode for free from the hardware -- see the comment on
// create_ldr_image() for why that is the correct domain -- but sRGB is not a
// mandatory blit destination, and img_in is a blit destination. Fall back to
// UNORM rather than failing: that is what this pass did before, so the
// resulting image is the older, worse one rather than no image at all.
static VkFormat pick_ldr_format(void)
{
	const VkFormatFeatureFlags required =
		  VK_FORMAT_FEATURE_BLIT_DST_BIT              // img_in: blitted into
		| VK_FORMAT_FEATURE_TRANSFER_SRC_BIT          // img_in: copied to buffer
		| VK_FORMAT_FEATURE_TRANSFER_DST_BIT          // img_out: copied from buffer
		| VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT         // img_out: sampled by the final blit
		| VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

	VkFormatProperties props;
	vkGetPhysicalDeviceFormatProperties(qvk.physical_device, VK_FORMAT_R8G8B8A8_SRGB, &props);

	if ((props.optimalTilingFeatures & required) == required)
		return VK_FORMAT_R8G8B8A8_SRGB;

	Com_WPrintf("upscaler: R8G8B8A8_SRGB is not usable for the model round trip on this "
		"device; running it in linear space instead, which is not what the model was "
		"trained on\n");
	return VK_FORMAT_R8G8B8A8_UNORM;
}

VkResult vkpt_upscaler_initialize(void)
{
	memset(&upscaler, 0, sizeof(upscaler));

	// Before any early return below: create_ldr_image() reads this
	// unconditionally, and VK_FORMAT_UNDEFINED from the memset would be invalid.
	upscaler.ldr_format = pick_ldr_format();

	upscaler.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (!upscaler.api) {
		Com_EPrintf("upscaler: failed to get ONNX Runtime API (expected ABI version %d)\n", ORT_API_VERSION);
		return VK_SUCCESS;
	}

	// Verbose severity is where ONNX Runtime reports per-node execution-provider
	// assignment, which is the authoritative signal that inference dispatches to
	// the Hexagon NPU rather than silently falling back to CPU. It's far too
	// chatty for normal play, so it's opt-in.
	OrtLoggingLevel log_level = (cvar_flt_upscaler_verbose && cvar_flt_upscaler_verbose->integer)
		? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING;

	if (!ort_ok(upscaler.api->CreateEnvWithCustomLogger(upscaler_ort_log, NULL, log_level,
			"q2rtx_upscaler", &upscaler.env), "CreateEnvWithCustomLogger"))
		return VK_SUCCESS;

	if (!ort_ok(upscaler.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &upscaler.cpu_memory_info), "CreateCpuMemoryInfo")) {
		upscaler.api->ReleaseEnv(upscaler.env);
		upscaler.env = NULL;
		return VK_SUCCESS;
	}

	Cmd_Register(upscaler_cmds);

	upscaler.initialized = true;

	// Picking a different model from the menu or the console drops the session;
	// init_cvars runs long before we exist, so the callback is only hooked up
	// here, and upscaler_reload_model() no-ops until initialized is set.
	cvar_flt_upscaler_enable->changed = upscaler_model_changed;
	upscaler.selected_model = selected_model_index();

	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy(void)
{
	unload_session();

	upscaler.initialized = false;
	cvar_flt_upscaler_enable->changed = NULL;
	Cmd_Deregister(upscaler_cmds);

	if (upscaler.cpu_memory_info) {
		upscaler.api->ReleaseMemoryInfo(upscaler.cpu_memory_info);
		upscaler.cpu_memory_info = NULL;
	}
	if (upscaler.env) {
		upscaler.api->ReleaseEnv(upscaler.env);
		upscaler.env = NULL;
	}

	return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Render integration
// ---------------------------------------------------------------------------

static void destroy_ldr_image(upscaler_image_t *img)
{
	if (img->view) {
		vkDestroyImageView(qvk.device, img->view, NULL);
		img->view = VK_NULL_HANDLE;
	}
	if (img->image) {
		vkDestroyImage(qvk.device, img->image, NULL);
		img->image = VK_NULL_HANDLE;
	}
	if (img->memory) {
		vkFreeMemory(qvk.device, img->memory, NULL);
		img->memory = VK_NULL_HANDLE;
	}
	img->extent = (VkExtent2D){ 0, 0 };
}

// The sRGB format is what puts the model in its training domain, and it costs
// nothing: the blit that fills img_in converts linear -> sRGB as part of its
// format conversion, and the sampler on img_out converts back on the way to the
// final blit, which owes the swapchain linear. The two cancel, so the displayed
// image is unchanged; what changes is that the 8 bits the tensor carries are
// spent perceptually rather than linearly.
//
// The copies in between are untouched by this: vkCmdCopyImageToBuffer and
// vkCmdCopyBufferToImage move raw bytes and never convert, which is exactly
// what the tensor wants.
static bool create_ldr_image(upscaler_image_t *img, VkExtent2D extent, const char *name)
{
	VkImageCreateInfo image_info = {
		.sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType             = VK_IMAGE_TYPE_2D,
		.format                = upscaler.ldr_format,
		.extent                = { extent.width, extent.height, 1 },
		.mipLevels             = 1,
		.arrayLayers           = 1,
		.samples               = VK_SAMPLE_COUNT_1_BIT,
		.tiling                = VK_IMAGE_TILING_OPTIMAL,
		.usage                 = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
		                       | VK_IMAGE_USAGE_TRANSFER_DST_BIT
		                       | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	if (vkCreateImage(qvk.device, &image_info, NULL, &img->image) != VK_SUCCESS) {
		Com_EPrintf("upscaler: failed to create %s image (%ux%u)\n", name, extent.width, extent.height);
		return false;
	}
	ATTACH_LABEL_VARIABLE_NAME(img->image, IMAGE, name);

	VkMemoryRequirements mem_req;
	vkGetImageMemoryRequirements(qvk.device, img->image, &mem_req);

	if (allocate_gpu_memory(mem_req, &img->memory) != VK_SUCCESS) {
		Com_EPrintf("upscaler: failed to allocate %.1f MB for the %s image\n",
			(double)mem_req.size / (1024.0 * 1024.0), name);
		destroy_ldr_image(img);
		return false;
	}

	if (vkBindImageMemory(qvk.device, img->image, img->memory, 0) != VK_SUCCESS) {
		destroy_ldr_image(img);
		return false;
	}

	VkImageViewCreateInfo view_info = {
		.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image            = img->image,
		.viewType         = VK_IMAGE_VIEW_TYPE_2D,
		.format           = upscaler.ldr_format,
		.subresourceRange = {
			.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel   = 0, .levelCount = 1,
			.baseArrayLayer = 0, .layerCount = 1,
		},
	};

	if (vkCreateImageView(qvk.device, &view_info, NULL, &img->view) != VK_SUCCESS) {
		destroy_ldr_image(img);
		return false;
	}

	img->extent = extent;
	return true;
}

// Releases one slot's staging buffers and images. Safe to call when they were
// never allocated.
static void destroy_slot(upscaler_slot_t *slot)
{
	if (slot->input_mapped) {
		buffer_unmap(&slot->buf_input);
		slot->input_mapped = NULL;
	}
	if (slot->output_mapped) {
		buffer_unmap(&slot->buf_output);
		slot->output_mapped = NULL;
	}

	buffer_destroy(&slot->buf_input);
	buffer_destroy(&slot->buf_output);

	destroy_ldr_image(&slot->img_in);
	destroy_ldr_image(&slot->img_out);

	slot->extent_in = (VkExtent2D){ 0, 0 };
	slot->extent_out = (VkExtent2D){ 0, 0 };
	slot->downloaded = false;
	slot->tensor_valid = false;
}

static void destroy_slots(void)
{
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		destroy_slot(&upscaler.slots[i]);
}

// (Re)allocates one slot's staging buffers and transfer images for the session's
// extent.
//
// Reallocating mid-frame is safe because only the downloading frame's own slot
// is touched, and the last GPU work to reference it was frame N-2's upload,
// which R_BeginFrame_RTX's fence wait has already accounted for.
static bool ensure_slot_resources(upscaler_slot_t *slot, VkExtent2D extent_in, VkExtent2D extent_out)
{
	if (upscaler_extents_equal(slot->extent_in, extent_in) &&
		upscaler_extents_equal(slot->extent_out, extent_out) && slot->input_mapped)
		return true;

	destroy_slot(slot);

	const VkMemoryPropertyFlags host_props =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// Adreno is a unified-memory part, so the transfers read/write these directly
	// and the CPU maps them persistently -- no separate device-local copy.
	if (buffer_create(&slot->buf_input, upscaler.input_byte_size,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_props) != VK_SUCCESS ||
		buffer_create(&slot->buf_output, upscaler.output_byte_size,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_props) != VK_SUCCESS)
	{
		Com_EPrintf("upscaler: failed to allocate %ux%u tensor buffers\n", extent_in.width, extent_in.height);
		destroy_slot(slot);
		return false;
	}

	buffer_attach_name(&slot->buf_input, "upscaler input tensor");
	buffer_attach_name(&slot->buf_output, "upscaler output tensor");

	slot->input_mapped  = buffer_map(&slot->buf_input);
	slot->output_mapped = buffer_map(&slot->buf_output);

	if (!slot->input_mapped || !slot->output_mapped) {
		Com_EPrintf("upscaler: failed to map tensor buffers\n");
		destroy_slot(slot);
		return false;
	}

	if (!create_ldr_image(&slot->img_in, extent_in, "upscaler input image") ||
		!create_ldr_image(&slot->img_out, extent_out, "upscaler output image"))
	{
		destroy_slot(slot);
		return false;
	}

	slot->extent_in = extent_in;
	slot->extent_out = extent_out;

	return true;
}

VkResult vkpt_upscaler_create_pipelines(void)
{
	// Nothing to compile any more -- the transfer is copy commands. All this
	// needs is the fences that track "the download reached the NPU's side".
	// Created unsignalled: every wait is gated on the slot's `downloaded` flag,
	// and nothing sets that until a submit has been given the fence to signal.
	VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (!upscaler.slots[i].download_fence)
			_VK(vkCreateFence(qvk.device, &fence_info, NULL, &upscaler.slots[i].download_fence));
	}
	vkpt_upscaler_discard();

	upscaler.ready = true;

	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy_pipelines(void)
{
	destroy_slots();
	reset_timing();

	upscaler.ready = false;

	// Every caller has already brought the device idle, so no fence destroyed
	// here can still be pending.
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		upscaler.slots[i].fence_pending = false;
		if (upscaler.slots[i].download_fence) {
			vkDestroyFence(qvk.device, upscaler.slots[i].download_fence, NULL);
			upscaler.slots[i].download_fence = VK_NULL_HANDLE;
		}
	}

	return VK_SUCCESS;
}

bool vkpt_upscaler_is_enabled(void)
{
	return spatial_is_active();
}

// The model's fixed scale factor, or 0 when nothing is going to run. This is a
// property of the loaded model, not a resolution policy: the render extent comes
// from viewsize/DRS like every other path, and the factor only says how much
// bigger than that the model's output will be.
uint32_t vkpt_upscaler_get_scale(void)
{
	return spatial_is_active() ? upscaler_models[upscaler.selected_model - 1].scale : 0;
}

// Brings the session in line with the render extent, rebuilding it when they
// disagree. Returns false when this frame cannot be upscaled.
static bool ensure_session(VkExtent2D extent)
{
	if (upscaler.session_ready && upscaler_extents_equal(upscaler.session_extent, extent))
		return true;

	if (upscaler.session_failed)
		return false;

	// Dynamic render scaling moves the render extent every frame, and every
	// distinct extent needs its own HTP graph. There is no rate at which that is
	// worth doing, so refuse outright rather than stalling for seconds at a time
	// to produce the occasional upscaled frame.
	if (cvar_drs_enable && cvar_drs_enable->integer) {
		upscaler.session_failed = true;
		Com_WPrintf("NPU upscaler: dynamic resolution scaling changes the render extent "
			"every frame, and each one costs a multi-second NPU graph rebuild. "
			"Disabling the upscaler; set drs_enable 0 and a fixed viewsize, then "
			"pick the upscaler again.\n");
		return false;
	}

	// Even with a fixed viewsize the extent can move in bursts -- dragging the
	// window, or holding down the viewsize slider. Each rebuild costs seconds, so
	// only let one through per interval and skip the pass in between.
	unsigned now = Sys_Milliseconds();
	if (upscaler.last_rebuild_ms != 0 && now - upscaler.last_rebuild_ms < UPSCALER_REBUILD_INTERVAL_MS)
		return false;

	if (upscaler.session_ready) {
		Com_Printf("upscaler: render extent changed to %ux%u, rebuilding the session\n",
			extent.width, extent.height);
	}

	if (!load_session(extent)) {
		// Do not retry until the selection changes: a missing or malformed model
		// fails identically every frame, and each attempt costs a QNN graph
		// finalization. This also turns spatial_is_active() off, so the render
		// extent goes back to the non-upscaled policy.
		upscaler.session_failed = true;
		upscaler.last_rebuild_ms = Sys_Milliseconds();
		Com_EPrintf("upscaler: disabling the NPU upscaler; "
			"set flt_upscaling again to retry\n");
		return false;
	}

	// Timed from after the rebuild, not before it: the interval is meant to space
	// out rebuilds, and the rebuild itself already took most of a second.
	upscaler.last_rebuild_ms = Sys_Milliseconds();
	return true;
}

// Copies the tone-mapped frame into `slot`'s input tensor, from
// vkpt_upscaler_do(). No shader: the blit converts rgba16f to unorm8 (clamping
// to [0,1] on the way, which is what the old pack shader did explicitly) and the
// copy lays the result out as the tightly packed NHWC RGBA the model wants.
static VkResult spatial_download(VkCommandBuffer cmd_buf, upscaler_slot_t *slot)
{
	slot->downloaded = false;

	VkExtent2D extent_in  = upscaler.session_extent;
	VkExtent2D extent_out = {
		extent_in.width  * upscaler_models[upscaler.selected_model - 1].scale,
		extent_in.height * upscaler_models[upscaler.selected_model - 1].scale,
	};

	if (!ensure_slot_resources(slot, extent_in, extent_out))
		return VK_SUCCESS;

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_DOWNLOAD);

	// vkpt_taa() only fills a top-left sub-rect of an image allocated at the
	// screen-image extent, and the session was built for exactly that sub-rect,
	// so both the blit and the copy work on extent_in rather than the whole image.
	IMAGE_BARRIER(cmd_buf,
		.image            = qvk.images[VKPT_IMG_TAA_OUTPUT],
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
		.oldLayout        = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout        = VK_IMAGE_LAYOUT_GENERAL,
	);

	// UNDEFINED as the old layout: the blit overwrites every texel, so there is
	// nothing to preserve and nothing to track across frames.
	IMAGE_BARRIER(cmd_buf,
		.image            = slot->img_in.image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		.srcAccessMask    = 0,
		.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	);

	VkImageBlit blit = {
		.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.srcOffsets     = { { 0, 0, 0 }, { (int32_t)extent_in.width, (int32_t)extent_in.height, 1 } },
		.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.dstOffsets     = { { 0, 0, 0 }, { (int32_t)extent_in.width, (int32_t)extent_in.height, 1 } },
	};
	vkCmdBlitImage(cmd_buf,
		qvk.images[VKPT_IMG_TAA_OUTPUT], VK_IMAGE_LAYOUT_GENERAL,
		slot->img_in.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &blit, VK_FILTER_NEAREST);

	IMAGE_BARRIER(cmd_buf,
		.image            = slot->img_in.image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		.srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
		.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	);

	// bufferRowLength/bufferImageHeight 0 means tightly packed to imageExtent,
	// which is exactly the tensor's stride.
	VkBufferImageCopy copy = {
		.bufferOffset      = 0,
		.bufferRowLength   = 0,
		.bufferImageHeight = 0,
		.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageOffset       = { 0, 0, 0 },
		.imageExtent       = { extent_in.width, extent_in.height, 1 },
	};
	vkCmdCopyImageToBuffer(cmd_buf, slot->img_in.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		slot->buf_input.buffer, 1, &copy);

	// Make the copy visible to the host read that next frame's
	// spatial_run_inference() will do once download_fence reports it done.
	BUFFER_BARRIER(cmd_buf,
		.buffer        = slot->buf_input.buffer,
		.offset        = 0,
		.size          = VK_WHOLE_SIZE,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_DOWNLOAD);
	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);

	// The fence is signalled by the frame's post submit, which is the command
	// buffer this pass records into -- see vkpt_upscaler_download_fence().
	//
	// It can still be pending here if the frame that would have waited on it
	// never ran (vkpt_upscaler_discard()), and resetting a pending fence is
	// invalid. Draining it first costs nothing: that submit is at least two
	// frames old, so R_BeginFrame_RTX's own fence wait has already covered it.
	if (slot->fence_pending)
		_VK(vkWaitForFences(qvk.device, 1, &slot->download_fence, VK_TRUE, ~((uint64_t)0)));
	_VK(vkResetFences(qvk.device, 1, &slot->download_fence));
	slot->fence_pending = true;

	slot->dump = upscaler.dump_requested;
	upscaler.dump_requested = false;

	slot->downloaded = true;

	return VK_SUCCESS;
}

// The fence the frame's post command buffer must signal, or VK_NULL_HANDLE when
// it carries no download -- spatial_download() bails on a failed allocation
// without recording one.
VkFence vkpt_upscaler_download_fence(void)
{
	upscaler_slot_t *slot = spatial_slot_cur();
	return slot->downloaded ? slot->download_fence : VK_NULL_HANDLE;
}

// Runs the NPU for the tensor `slot` holds, which -- because the round trip is
// pipelined -- is the *previous* frame's. No queue drain: the download's own
// submit signalled download_fence, and a whole frame of GPU work has happened
// since, so this wait is usually already satisfied. Where it is not, it blocks
// only on the previous frame's post pass rather than on everything submitted so
// far, and the GPU stays free to work through the current frame.
static VkResult spatial_run_inference(upscaler_slot_t *slot)
{
	if (!slot->downloaded)
		return VK_SUCCESS;

	slot->downloaded = false;
	slot->tensor_valid = false;

	bool dump = slot->dump;
	slot->dump = false;

	unsigned stall_begin = Sys_Milliseconds();

	_VK(vkWaitForFences(qvk.device, 1, &slot->download_fence, VK_TRUE, ~((uint64_t)0)));
	slot->fence_pending = false;

	unsigned stall_end = Sys_Milliseconds();

	// The session can have been rebuilt at a different extent between the
	// download and now, which leaves this tensor the wrong size for it.
	if (!upscaler.session_ready || !upscaler_extents_equal(slot->extent_in, upscaler.session_extent))
		return VK_SUCCESS;

	// Outside the timed region below: writing a multi-megabyte PNG dwarfs the
	// inference and would otherwise land entirely in this frame's sample.
	if (dump)
		upscaler_dump_tensor("pre", (const uint8_t *)slot->input_mapped, slot->extent_in);

	unsigned infer_begin = Sys_Milliseconds();

	const char *input_names[]  = { UPSCALER_INPUT_NAME };
	const char *output_names[] = { UPSCALER_OUTPUT_NAME };

	OrtValue *input_value = NULL, *output_value = NULL;

	// Wrapping the mapped staging memory directly avoids an extra copy on each
	// side of the inference.
	if (ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, slot->input_mapped,
			upscaler.input_byte_size, upscaler.input_dims, UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &input_value), "CreateTensorWithDataAsOrtValue(input)") &&
		ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, slot->output_mapped,
			upscaler.output_byte_size, upscaler.output_dims, UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &output_value), "CreateTensorWithDataAsOrtValue(output)"))
	{
		const OrtValue *inputs[] = { input_value };
		OrtValue *outputs[]      = { output_value };

		slot->tensor_valid = ort_ok(upscaler.api->Run(upscaler.session, NULL,
			input_names, inputs, 1, output_names, 1, outputs), "inference");
	}

	if (input_value)
		upscaler.api->ReleaseValue(input_value);
	if (output_value)
		upscaler.api->ReleaseValue(output_value);

	// The ORT tensor create/release pairs above are inside the measurement on
	// purpose -- they are real per-frame CPU cost, and they are what the game
	// reads above a bare Run() benchmark.
	unsigned infer_end = Sys_Milliseconds();

	if (dump && slot->tensor_valid)
		upscaler_dump_tensor("post", (const uint8_t *)slot->output_mapped, slot->extent_out);

	upscaler.stall_ms_accum += stall_end - stall_begin;
	upscaler.inference_ms_accum += infer_end - infer_begin;
	upscaler.inference_frames++;

	if (upscaler.inference_frames >= UPSCALER_TIMING_INTERVAL)
	{
		// Naming the model as well as the extent: reset_timing() guarantees every
		// sample in this average came from one session, so the label is exact.
		Com_Printf("upscaler: %s at %ux%u: %.2f ms/frame inference, "
			"%.2f ms/frame blocked on the previous frame's GPU work\n",
			upscaler_models[upscaler.selected_model - 1].name,
			slot->extent_in.width, slot->extent_in.height,
			(double)upscaler.inference_ms_accum / upscaler.inference_frames,
			(double)upscaler.stall_ms_accum / upscaler.inference_frames);
		reset_timing();
	}

	return VK_SUCCESS;
}

// The frame's NPU work: the inference for whatever the previous frame
// downloaded, then this frame's download. Running the inference here, before
// anything is recorded, gives download_fence the whole preceding frame to be
// signalled in.
VkResult vkpt_upscaler_do(VkCommandBuffer cmd_buf)
{
	if (!spatial_is_active())
		return VK_SUCCESS;

	spatial_run_inference(spatial_slot_prev());

	if (!ensure_session(qvk.extent_taa_output))
		return VK_SUCCESS;

	return spatial_download(cmd_buf, spatial_slot_cur());
}

VkResult vkpt_upscaler_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	// One frame behind: the tensor this uploads was downloaded last frame and
	// inferred at the top of this one. A slot's result is good for exactly one
	// blit, so take it and clear it.
	upscaler_slot_t *slot = spatial_slot_prev();
	bool tensor_valid = slot->tensor_valid;
	slot->tensor_valid = false;

	// The pass can bail before producing anything -- a failed staging allocation
	// leaves the buffers destroyed, a failed inference leaves the tensor holding
	// an older frame, and on a cold start there is no previous frame at all.
	// Presenting any of those would show garbage or read an already-freed
	// buffer, so fall back to the tone-mapped frame, which is below display
	// resolution here, hence the filtered blit.
	if (!tensor_valid) {
		bool needs_filter = qvk.extent_taa_output.width  != qvk.extent_unscaled.width
		                 || qvk.extent_taa_output.height != qvk.extent_unscaled.height;
		return vkpt_final_blit(cmd_buf, VKPT_IMG_TAA_OUTPUT, qvk.extent_taa_output,
			needs_filter, warp);
	}

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UPLOAD);

	// No host-write barrier: the inference wrote buf_output before this command
	// buffer was submitted, and vkQueueSubmit makes host writes visible to the
	// device automatically.
	IMAGE_BARRIER(cmd_buf,
		.image            = slot->img_out.image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		.srcAccessMask    = 0,
		.dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	);

	VkBufferImageCopy copy = {
		.bufferOffset      = 0,
		.bufferRowLength   = 0,
		.bufferImageHeight = 0,
		.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageOffset       = { 0, 0, 0 },
		.imageExtent       = { slot->extent_out.width, slot->extent_out.height, 1 },
	};
	vkCmdCopyBufferToImage(cmd_buf, slot->buf_output.buffer, slot->img_out.image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

	IMAGE_BARRIER(cmd_buf,
		.image            = slot->img_out.image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		.srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.newLayout        = VK_IMAGE_LAYOUT_GENERAL,
	);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UPLOAD);

	// The model always multiplies by its fixed factor, so the result overshoots
	// the display by viewsize * scale / 100 and the blit resolves the excess.
	// At viewsize 100 / scale the two agree and this is a 1:1 copy.
	bool needs_filter = !upscaler_extents_equal(slot->extent_out, qvk.extent_unscaled);

	return vkpt_final_blit_view(cmd_buf, slot->img_out.view, slot->extent_out, needs_filter, warp);
}

#else // !USE_ORT_QNN_UPSCALER

// No ONNX Runtime in this build, so there is nothing to swap; flt_upscaling
// still multiplexes correctly, the AI entries just never become available.
static void upscaler_reload_model(void)
{
}

VkResult vkpt_upscaler_initialize(void)
{
	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy(void)
{
	return VK_SUCCESS;
}

VkResult vkpt_upscaler_create_pipelines(void)
{
	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy_pipelines(void)
{
	return VK_SUCCESS;
}

bool vkpt_upscaler_is_enabled(void)
{
	return false;
}

uint32_t vkpt_upscaler_get_scale(void)
{
	return 0;
}

VkResult vkpt_upscaler_do(VkCommandBuffer cmd_buf)
{
	return VK_SUCCESS;
}

VkResult vkpt_upscaler_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	return VK_SUCCESS;
}

VkFence vkpt_upscaler_download_fence(void)
{
	return VK_NULL_HANDLE;
}

void vkpt_upscaler_discard(void)
{
}

#endif // USE_ORT_QNN_UPSCALER

// vim: shiftwidth=4 noexpandtab tabstop=4 cindent
