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

	Phase 1 (current): loads QuickSRNetSmall (w8a8, ONNX) through ONNX Runtime's
	Qualcomm QNN Execution Provider and proves inference actually dispatches to
	the Hexagon NPU on this Windows-ARM64 (Snapdragon) device, via a standalone
	console-command-triggered test. There is no render-pipeline integration yet:
	vkpt_upscaler_do()/_final_blit() are no-ops and no Vulkan resources are
	created. That integration (pack/unpack compute shaders, GPU<->NPU staging
	buffers, wiring into R_RenderFrame_RTX/R_EndFrame_RTX in place of FSR) is
	Phase 2, tracked separately.

	The model's input/output tensors are a static NCHW uint8 shape (1x3x128x128
	in, 1x3x512x512 out, i.e. a fixed 4x upscale) -- not dynamic -- so Phase 2's
	shaders will need to tile the actual frame into 128x128 blocks.

	Q2RTX cvars
	-----------
	* flt_upscaler_enable - 0 = disable, 1 = enable the NPU upscaler.

	Console commands
	----------------
	* upscaler_npu_test [seconds] - run repeated inference for the given
	  duration (default 3s) so NPU dispatch/utilization can be confirmed via
	  Windows Task Manager > Performance > NPU while it runs, and via the
	  verbose ONNX Runtime log lines this prints to the console.
*/

#include "shared/shared.h"
#include "common/common.h"
#include "vkpt.h"

#include <string.h>
#include <stdlib.h>

cvar_t *cvar_flt_upscaler_enable = NULL;
cvar_t *cvar_flt_upscaler_verbose = NULL;

extern cvar_t *cvar_flt_fsr_enable; // owned by fsr.c, initialized just before us

cvar_t *cvar_flt_upscaling = NULL;

// Menu-facing selector for the mutually exclusive upscalers. The per-backend
// cvars stay authoritative so existing configs, scripts and console use keep
// working; this just keeps them from being enabled at the same time.
enum {
	UPSCALING_MODE_NONE = 0,
	UPSCALING_MODE_FSR  = 1,
	UPSCALING_MODE_AI   = 2,
};

static void upscaling_mode_changed(cvar_t *self)
{
	Cvar_SetInteger(cvar_flt_fsr_enable, self->integer == UPSCALING_MODE_FSR, FROM_CODE);
	Cvar_SetInteger(cvar_flt_upscaler_enable, self->integer == UPSCALING_MODE_AI, FROM_CODE);
}

void vkpt_upscaler_init_cvars(void)
{
	cvar_flt_upscaler_enable = Cvar_Get("flt_upscaler_enable", "0", CVAR_ARCHIVE);
	// Dumps ONNX Runtime's per-node execution-provider assignment at startup,
	// which is how you confirm the model is really running on the NPU.
	cvar_flt_upscaler_verbose = Cvar_Get("flt_upscaler_verbose", "0", 0);

	// Seeded from whatever the backend cvars already say, so a config that set
	// flt_fsr_enable directly shows up correctly in the menu.
	const char *initial = cvar_flt_upscaler_enable->integer ? "2"
		: (cvar_flt_fsr_enable && cvar_flt_fsr_enable->integer ? "1" : "0");

	cvar_flt_upscaling = Cvar_Get("flt_upscaling", initial, CVAR_ARCHIVE);
	cvar_flt_upscaling->changed = upscaling_mode_changed;
	upscaling_mode_changed(cvar_flt_upscaling);
}

#ifdef USE_ORT_QNN_UPSCALER

#include <windows.h>
#include "onnxruntime_c_api.h"
#include "vk_util.h"
#include "shader/upscaler_shared.h"

#define UPSCALER_MODEL_PATH "models/quicksrnetsmall-w8a8.onnx"
#define UPSCALER_INPUT_NAME "image"
#define UPSCALER_OUTPUT_NAME "upscaled_image"
#define UPSCALER_NUM_DIMS 4
#define UPSCALER_DEFAULT_TEST_SECONDS 3

// Per-tile tensor sizes, in bytes (uint8, 3 channels, NCHW).
#define UPSCALER_TILE_IN_BYTES  (3u * UPSCALER_TILE_IN  * UPSCALER_TILE_IN)
#define UPSCALER_TILE_OUT_BYTES (3u * UPSCALER_TILE_OUT * UPSCALER_TILE_OUT)

typedef struct {
	uint32_t tiles_x;
	uint32_t tiles_y;
} upscaler_push_constants_t;

struct
{
	const OrtApi  *api;
	OrtEnv        *env;
	OrtSession    *session;
	OrtMemoryInfo *cpu_memory_info;
	bool           model_loaded;

	int64_t        input_dims[UPSCALER_NUM_DIMS];
	int64_t        output_dims[UPSCALER_NUM_DIMS];
	size_t         input_byte_size;  // uint8 tensor, 1 byte/element
	size_t         output_byte_size;

	// Phase 2 render integration.
	bool                  pipelines_ready;
	VkPipeline            pipeline_pack;
	VkPipeline            pipeline_unpack;
	VkPipelineLayout      pipeline_layout;
	VkDescriptorSetLayout desc_set_layout;
	VkDescriptorPool      desc_pool;
	VkDescriptorSet       desc_set;

	// Host-visible staging for the GPU <-> NPU round trip. Adreno is a
	// unified-memory part, so the compute shaders read/write these directly
	// and the CPU maps them persistently -- no separate device-local copy.
	BufferResource_t buf_input;
	BufferResource_t buf_output;
	void            *input_mapped;
	void            *output_mapped;
	uint32_t         tiles_x;
	uint32_t         tiles_y;

	// Set by vkpt_upscaler_do(), consumed by vkpt_upscaler_run_inference().
	bool             packed_this_frame;

	// Rolling cost of the NPU round trip. Accumulated over many frames because
	// Sys_Milliseconds() is far too coarse to time a single frame's inference.
	unsigned         inference_ms_accum;
	unsigned         inference_frames;
} upscaler;

#define UPSCALER_TIMING_INTERVAL 100 // frames between timing reports

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

static bool query_tensor_shape(bool is_input, size_t index, const char *expected_name,
	int64_t *dims_out, size_t *byte_size_out)
{
	OrtAllocator *allocator;
	if (!ort_ok(upscaler.api->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions"))
		return false;

	char *name = NULL;
	OrtTypeInfo *type_info = NULL;
	bool ok = false;

	OrtStatus *status = is_input
		? upscaler.api->SessionGetInputName(upscaler.session, index, allocator, &name)
		: upscaler.api->SessionGetOutputName(upscaler.session, index, allocator, &name);
	if (!ort_ok(status, "SessionGetInput/OutputName"))
		return false;

	if (strcmp(name, expected_name) != 0) {
		Com_WPrintf("upscaler: model %s %zu is named '%s', expected '%s'\n",
			is_input ? "input" : "output", index, name, expected_name);
	}

	status = is_input
		? upscaler.api->SessionGetInputTypeInfo(upscaler.session, index, &type_info)
		: upscaler.api->SessionGetOutputTypeInfo(upscaler.session, index, &type_info);
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
			Com_EPrintf("upscaler: expected a uint8 tensor, model %s %zu has element type %d\n",
				is_input ? "input" : "output", index, (int)elem_type);
			goto done;
		}

		size_t element_count = 1;
		for (size_t i = 0; i < num_dims; i++)
			element_count *= (size_t)dims_out[i];
		*byte_size_out = element_count;

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
// timing can be sanity-checked against Qualcomm's published benchmarks
// (~0.5ms for QuickSRNetSmall w8a8 on Snapdragon X Elite).
static void Upscaler_NpuTest_f(void)
{
	if (!upscaler.model_loaded) {
		Com_Printf("upscaler: NPU upscaler model not loaded, nothing to test\n");
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

		Com_Printf("upscaler: running NPU test inference for %d ms "
			"(watch Task Manager > Performance > NPU)\n", duration_ms);

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

static void destroy_session(void)
{
	if (upscaler.cpu_memory_info) {
		upscaler.api->ReleaseMemoryInfo(upscaler.cpu_memory_info);
		upscaler.cpu_memory_info = NULL;
	}
	if (upscaler.session) {
		upscaler.api->ReleaseSession(upscaler.session);
		upscaler.session = NULL;
	}
	if (upscaler.env) {
		upscaler.api->ReleaseEnv(upscaler.env);
		upscaler.env = NULL;
	}
}

VkResult vkpt_upscaler_initialize(void)
{
	memset(&upscaler, 0, sizeof(upscaler));

	upscaler.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (!upscaler.api) {
		Com_EPrintf("upscaler: failed to get ONNX Runtime API (expected ABI version %d)\n", ORT_API_VERSION);
		return VK_SUCCESS;
	}

	// Verbose severity is where ONNX Runtime reports per-node execution-provider
	// assignment, which is the authoritative signal that inference dispatches to
	// the Hexagon NPU rather than silently falling back to CPU. It's far too
	// chatty for normal play (it spams the console every frame), so it's opt-in.
	OrtLoggingLevel log_level = (cvar_flt_upscaler_verbose && cvar_flt_upscaler_verbose->integer)
		? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING;

	if (!ort_ok(upscaler.api->CreateEnvWithCustomLogger(upscaler_ort_log, NULL, log_level,
			"q2rtx_upscaler", &upscaler.env), "CreateEnvWithCustomLogger"))
		return VK_SUCCESS;

	OrtSessionOptions *session_options = NULL;
	if (!ort_ok(upscaler.api->CreateSessionOptions(&session_options), "CreateSessionOptions"))
		goto fail;

	{
		const char *provider_keys[]   = { "backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode" };
		const char *provider_values[] = { "QnnHtp.dll", "high_performance", "3" };
		OrtStatus *ep_status = upscaler.api->SessionOptionsAppendExecutionProvider(session_options, "QNN",
			provider_keys, provider_values, LENGTH(provider_keys));
		if (!ort_ok(ep_status, "SessionOptionsAppendExecutionProvider(QNN)")) {
			Com_Printf("upscaler: QNN execution provider unavailable, NPU upscaler disabled\n");
			upscaler.api->ReleaseSessionOptions(session_options);
			goto fail;
		}
	}

	{
		char model_path[MAX_OSPATH];
		if (Q_concat(model_path, sizeof(model_path), fs_gamedir, PATH_SEP_STRING, UPSCALER_MODEL_PATH) >= sizeof(model_path)) {
			Com_EPrintf("upscaler: model path too long\n");
			upscaler.api->ReleaseSessionOptions(session_options);
			goto fail;
		}

		WCHAR wmodel_path[MAX_OSPATH];
		MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wmodel_path, MAX_OSPATH);

		OrtStatus *session_status = upscaler.api->CreateSession(upscaler.env, wmodel_path, session_options, &upscaler.session);
		upscaler.api->ReleaseSessionOptions(session_options);

		if (!ort_ok(session_status, "CreateSession")) {
			Com_Printf("upscaler: could not load %s, NPU upscaler unavailable\n", model_path);
			goto fail;
		}
	}

	if (!query_tensor_shape(true, 0, UPSCALER_INPUT_NAME, upscaler.input_dims, &upscaler.input_byte_size))
		goto fail;
	if (!query_tensor_shape(false, 0, UPSCALER_OUTPUT_NAME, upscaler.output_dims, &upscaler.output_byte_size))
		goto fail;

	// The pack/unpack shaders hardcode the tile geometry (upscaler_shared.h),
	// so a model with different fixed dimensions would silently corrupt the
	// tensor layout. Refuse it rather than render garbage.
	if (upscaler.input_byte_size != UPSCALER_TILE_IN_BYTES ||
		upscaler.output_byte_size != UPSCALER_TILE_OUT_BYTES)
	{
		Com_EPrintf("upscaler: model tensors are %zu->%zu bytes, expected %ux%u->%ux%u RGB8 "
			"(%u->%u bytes); NPU upscaler disabled\n",
			upscaler.input_byte_size, upscaler.output_byte_size,
			UPSCALER_TILE_IN, UPSCALER_TILE_IN, UPSCALER_TILE_OUT, UPSCALER_TILE_OUT,
			UPSCALER_TILE_IN_BYTES, UPSCALER_TILE_OUT_BYTES);
		goto fail;
	}

	if (!ort_ok(upscaler.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &upscaler.cpu_memory_info), "CreateCpuMemoryInfo"))
		goto fail;

	upscaler.model_loaded = true;

	{
		cmdreg_t cmds[] = {
			{ "upscaler_npu_test", &Upscaler_NpuTest_f, NULL },
			{ NULL, NULL, NULL }
		};
		Cmd_Register(cmds);
	}

	Com_Printf("upscaler: NPU upscaler model loaded (%s); run 'upscaler_npu_test' to verify NPU dispatch\n", UPSCALER_MODEL_PATH);
	return VK_SUCCESS;

fail:
	destroy_session();
	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy(void)
{
	destroy_session();
	upscaler.model_loaded = false;
	return VK_SUCCESS;
}

#define BARRIER_COMPUTE(cmd_buf, img) \
	do { \
		VkImageSubresourceRange subresource_range = { \
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, \
			.baseMipLevel = 0, .levelCount = 1, \
			.baseArrayLayer = 0, .layerCount = 1 }; \
		IMAGE_BARRIER(cmd_buf, \
				.image            = img, \
				.subresourceRange = subresource_range, \
				.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT, \
				.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT, \
				.oldLayout        = VK_IMAGE_LAYOUT_GENERAL, \
				.newLayout        = VK_IMAGE_LAYOUT_GENERAL, \
		); \
	} while(0)

// Releases the staging buffers. Safe to call when they were never allocated.
static void destroy_tensor_buffers(void)
{
	if (upscaler.input_mapped) {
		buffer_unmap(&upscaler.buf_input);
		upscaler.input_mapped = NULL;
	}
	if (upscaler.output_mapped) {
		buffer_unmap(&upscaler.buf_output);
		upscaler.output_mapped = NULL;
	}

	buffer_destroy(&upscaler.buf_input);
	buffer_destroy(&upscaler.buf_output);

	upscaler.tiles_x = 0;
	upscaler.tiles_y = 0;
}

// (Re)allocates the tensor staging buffers for the current tile grid and points
// the descriptor set at them. The grid is derived from the output image size,
// so this also covers resolution changes and dynamic render scaling.
static bool ensure_tensor_buffers(uint32_t tiles_x, uint32_t tiles_y)
{
	if (upscaler.tiles_x == tiles_x && upscaler.tiles_y == tiles_y && upscaler.input_mapped)
		return true;

	destroy_tensor_buffers();

	uint32_t num_tiles = tiles_x * tiles_y;
	VkDeviceSize input_size  = (VkDeviceSize)num_tiles * UPSCALER_TILE_IN_BYTES;
	VkDeviceSize output_size = (VkDeviceSize)num_tiles * UPSCALER_TILE_OUT_BYTES;

	const VkMemoryPropertyFlags host_props =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	if (buffer_create(&upscaler.buf_input, input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS ||
		buffer_create(&upscaler.buf_output, output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS)
	{
		Com_EPrintf("upscaler: failed to allocate %ux%u tile tensor buffers\n", tiles_x, tiles_y);
		destroy_tensor_buffers();
		return false;
	}

	buffer_attach_name(&upscaler.buf_input, "upscaler input tensor");
	buffer_attach_name(&upscaler.buf_output, "upscaler output tensor");

	upscaler.input_mapped  = buffer_map(&upscaler.buf_input);
	upscaler.output_mapped = buffer_map(&upscaler.buf_output);

	if (!upscaler.input_mapped || !upscaler.output_mapped) {
		Com_EPrintf("upscaler: failed to map tensor buffers\n");
		destroy_tensor_buffers();
		return false;
	}

	VkDescriptorBufferInfo buffer_info[] = {
		{ .buffer = upscaler.buf_input.buffer,  .offset = 0, .range = input_size  },
		{ .buffer = upscaler.buf_output.buffer, .offset = 0, .range = output_size },
	};

	VkWriteDescriptorSet writes[] = {
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = upscaler.desc_set,
			.dstBinding      = 0,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo     = &buffer_info[0],
		},
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = upscaler.desc_set,
			.dstBinding      = 1,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo     = &buffer_info[1],
		},
	};
	vkUpdateDescriptorSets(qvk.device, LENGTH(writes), writes, 0, NULL);

	upscaler.tiles_x = tiles_x;
	upscaler.tiles_y = tiles_y;

	Com_Printf("upscaler: %ux%u tile grid (%u inferences/frame, %.1f MB staging)\n",
		tiles_x, tiles_y, num_tiles, (double)(input_size + output_size) / (1024.0 * 1024.0));

	return true;
}

VkResult vkpt_upscaler_create_pipelines(void)
{
	if (!upscaler.model_loaded)
		return VK_SUCCESS;

	VkDescriptorSetLayoutBinding bindings[] = {
		{
			.binding         = 0, // input tensor, written by the pack pass
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
		},
		{
			.binding         = 1, // output tensor, read by the unpack pass
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
		},
	};

	VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = LENGTH(bindings),
		.pBindings    = bindings,
	};
	_VK(vkCreateDescriptorSetLayout(qvk.device, &layout_info, NULL, &upscaler.desc_set_layout));
	ATTACH_LABEL_VARIABLE(upscaler.desc_set_layout, DESCRIPTOR_SET_LAYOUT);

	VkDescriptorPoolSize pool_size = {
		.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = LENGTH(bindings),
	};
	VkDescriptorPoolCreateInfo pool_info = {
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets       = 1,
		.poolSizeCount = 1,
		.pPoolSizes    = &pool_size,
	};
	_VK(vkCreateDescriptorPool(qvk.device, &pool_info, NULL, &upscaler.desc_pool));

	VkDescriptorSetAllocateInfo alloc_info = {
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool     = upscaler.desc_pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &upscaler.desc_set_layout,
	};
	_VK(vkAllocateDescriptorSets(qvk.device, &alloc_info, &upscaler.desc_set));

	// Sets 0/1 match the convention the other post-processing compute shaders
	// use (see fsr.c); set 2 carries the tensor buffers.
	VkDescriptorSetLayout desc_set_layouts[] = {
		qvk.desc_set_layout_ubo,
		qvk.desc_set_layout_textures,
		upscaler.desc_set_layout,
	};

	VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset     = 0,
		.size       = sizeof(upscaler_push_constants_t),
	};

	CREATE_PIPELINE_LAYOUT(qvk.device, &upscaler.pipeline_layout,
		.setLayoutCount         = LENGTH(desc_set_layouts),
		.pSetLayouts            = desc_set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &push_constant_range,
	);
	ATTACH_LABEL_VARIABLE(upscaler.pipeline_layout, PIPELINE_LAYOUT);

	VkComputePipelineCreateInfo pipeline_info[] = {
		{
			.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage  = SHADER_STAGE(QVK_MOD_UPSCALER_PACK_COMP, VK_SHADER_STAGE_COMPUTE_BIT),
			.layout = upscaler.pipeline_layout,
		},
		{
			.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage  = SHADER_STAGE(QVK_MOD_UPSCALER_UNPACK_COMP, VK_SHADER_STAGE_COMPUTE_BIT),
			.layout = upscaler.pipeline_layout,
		},
	};

	VkPipeline pipelines[LENGTH(pipeline_info)];
	_VK(vkCreateComputePipelines(qvk.device, 0, LENGTH(pipeline_info), pipeline_info, 0, pipelines));

	upscaler.pipeline_pack   = pipelines[0];
	upscaler.pipeline_unpack = pipelines[1];
	upscaler.pipelines_ready = true;

	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy_pipelines(void)
{
	destroy_tensor_buffers();

	upscaler.pipelines_ready = false;

	if (upscaler.pipeline_pack) {
		vkDestroyPipeline(qvk.device, upscaler.pipeline_pack, NULL);
		upscaler.pipeline_pack = VK_NULL_HANDLE;
	}
	if (upscaler.pipeline_unpack) {
		vkDestroyPipeline(qvk.device, upscaler.pipeline_unpack, NULL);
		upscaler.pipeline_unpack = VK_NULL_HANDLE;
	}
	if (upscaler.pipeline_layout) {
		vkDestroyPipelineLayout(qvk.device, upscaler.pipeline_layout, NULL);
		upscaler.pipeline_layout = VK_NULL_HANDLE;
	}
	if (upscaler.desc_pool) {
		vkDestroyDescriptorPool(qvk.device, upscaler.desc_pool, NULL);
		upscaler.desc_pool = VK_NULL_HANDLE;
		upscaler.desc_set = VK_NULL_HANDLE;
	}
	if (upscaler.desc_set_layout) {
		vkDestroyDescriptorSetLayout(qvk.device, upscaler.desc_set_layout, NULL);
		upscaler.desc_set_layout = VK_NULL_HANDLE;
	}

	return VK_SUCCESS;
}

bool vkpt_upscaler_is_enabled(void)
{
	return cvar_flt_upscaler_enable->integer != 0
		&& upscaler.model_loaded
		&& upscaler.pipelines_ready;
}

static void bind_upscaler_pipeline(VkCommandBuffer cmd_buf, VkPipeline pipeline)
{
	VkDescriptorSet desc_sets[] = {
		qvk.desc_set_ubo,
		qvk_get_current_desc_set_textures(),
		upscaler.desc_set,
	};

	vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
		upscaler.pipeline_layout, 0, LENGTH(desc_sets), desc_sets, 0, NULL);

	upscaler_push_constants_t push = { upscaler.tiles_x, upscaler.tiles_y };
	vkCmdPushConstants(cmd_buf, upscaler.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
		0, sizeof(push), &push);
}

VkResult vkpt_upscaler_do(VkCommandBuffer cmd_buf)
{
	upscaler.packed_this_frame = false;

	// One tile of model output covers UPSCALER_TILE_OUT pixels of the final
	// image; the grid is sized to cover it and the edges are cropped on unpack.
	VkExtent2D out = qvk.extent_screen_images;
	uint32_t tiles_x = (out.width  + UPSCALER_TILE_OUT - 1) / UPSCALER_TILE_OUT;
	uint32_t tiles_y = (out.height + UPSCALER_TILE_OUT - 1) / UPSCALER_TILE_OUT;

	if (tiles_x == 0 || tiles_y == 0)
		return VK_SUCCESS;

	if (!ensure_tensor_buffers(tiles_x, tiles_y))
		return VK_SUCCESS;

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_PACK);

	bind_upscaler_pipeline(cmd_buf, upscaler.pipeline_pack);

	// The pack shader writes 4 horizontally adjacent pixels per invocation.
	uint32_t dispatch_x = tiles_x * (UPSCALER_TILE_IN / 4);
	uint32_t dispatch_y = tiles_y * UPSCALER_TILE_IN;
	vkCmdDispatch(cmd_buf, (dispatch_x + 7) / 8, (dispatch_y + 7) / 8, 1);

	// Make the shader writes visible to the host read in run_inference().
	BUFFER_BARRIER(cmd_buf,
		.buffer        = upscaler.buf_input.buffer,
		.offset        = 0,
		.size          = VK_WHOLE_SIZE,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_PACK);
	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);

	upscaler.packed_this_frame = true;

	return VK_SUCCESS;
}

// Runs the NPU inference for every tile of the frame that vkpt_upscaler_do()
// packed. Must be called after the command buffer containing the pack dispatch
// has been submitted -- see the call site in R_RenderFrame_RTX.
//
// This is a hard pipeline stall: the GPU has to finish the pack pass before the
// CPU can read the tensor, and the unpack pass can't be recorded until the NPU
// has produced the output. That round trip is inherent to driving the NPU from
// the CPU, and is the main reason per-frame NPU upscaling is only viable when
// the tile count stays low.
VkResult vkpt_upscaler_run_inference(void)
{
	if (!upscaler.packed_this_frame || !vkpt_upscaler_is_enabled())
		return VK_SUCCESS;

	upscaler.packed_this_frame = false;

	vkQueueWaitIdle(qvk.queue_graphics);

	unsigned time_begin = Sys_Milliseconds();

	uint32_t num_tiles = upscaler.tiles_x * upscaler.tiles_y;
	const char *input_names[]  = { UPSCALER_INPUT_NAME };
	const char *output_names[] = { UPSCALER_OUTPUT_NAME };

	for (uint32_t tile = 0; tile < num_tiles; tile++)
	{
		uint8_t *in  = (uint8_t *)upscaler.input_mapped  + (size_t)tile * UPSCALER_TILE_IN_BYTES;
		uint8_t *out = (uint8_t *)upscaler.output_mapped + (size_t)tile * UPSCALER_TILE_OUT_BYTES;

		OrtValue *input_value = NULL, *output_value = NULL;

		// Wrapping the mapped staging memory directly avoids an extra copy on
		// each side of the inference.
		if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, in,
				UPSCALER_TILE_IN_BYTES, upscaler.input_dims, UPSCALER_NUM_DIMS,
				ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &input_value), "CreateTensorWithDataAsOrtValue(input)"))
			break;

		if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, out,
				UPSCALER_TILE_OUT_BYTES, upscaler.output_dims, UPSCALER_NUM_DIMS,
				ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &output_value), "CreateTensorWithDataAsOrtValue(output)"))
		{
			upscaler.api->ReleaseValue(input_value);
			break;
		}

		const OrtValue *inputs[] = { input_value };
		OrtValue *outputs[]      = { output_value };

		OrtStatus *status = upscaler.api->Run(upscaler.session, NULL,
			input_names, inputs, 1, output_names, 1, outputs);

		upscaler.api->ReleaseValue(input_value);
		upscaler.api->ReleaseValue(output_value);

		if (status) {
			Com_EPrintf("upscaler: inference failed on tile %u: %s\n",
				tile, upscaler.api->GetErrorMessage(status));
			upscaler.api->ReleaseStatus(status);
			break;
		}
	}

	upscaler.inference_ms_accum += Sys_Milliseconds() - time_begin;
	upscaler.inference_frames++;

	if (upscaler.inference_frames >= UPSCALER_TIMING_INTERVAL)
	{
		Com_Printf("upscaler: %.2f ms/frame for %u tiles (%.2f ms/tile)\n",
			(double)upscaler.inference_ms_accum / upscaler.inference_frames,
			num_tiles,
			(double)upscaler.inference_ms_accum / (upscaler.inference_frames * num_tiles));
		upscaler.inference_ms_accum = 0;
		upscaler.inference_frames = 0;
	}

	return VK_SUCCESS;
}

VkResult vkpt_upscaler_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);

	bind_upscaler_pipeline(cmd_buf, upscaler.pipeline_unpack);

	VkExtent2D out = qvk.extent_screen_images;
	vkCmdDispatch(cmd_buf, (out.width + 7) / 8, (out.height + 7) / 8, 1);

	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_UPSCALE_OUTPUT]);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);

	return vkpt_final_blit(cmd_buf, VKPT_IMG_UPSCALE_OUTPUT, qvk.extent_unscaled, false, warp);
}

#else // !USE_ORT_QNN_UPSCALER

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

VkResult vkpt_upscaler_run_inference(void)
{
	return VK_SUCCESS;
}

VkResult vkpt_upscaler_do(VkCommandBuffer cmd_buf)
{
	return VK_SUCCESS;
}

VkResult vkpt_upscaler_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	return VK_SUCCESS;
}

#endif // USE_ORT_QNN_UPSCALER

// vim: shiftwidth=4 noexpandtab tabstop=4 cindent
