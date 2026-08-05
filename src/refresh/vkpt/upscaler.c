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
	NSS temporal upscaler ("upscaler") implementation overview
	==========================================================

	Runs Arm's Neural Super Sampling temporal model (int8 QAT export, ONNX)
	through ONNX Runtime's Qualcomm QNN Execution Provider, so inference
	dispatches to the Hexagon NPU on Windows-ARM64 (Snapdragon) devices.

	Per frame: nss_pack.comp builds one fixed-shape 12-channel float32 NCHW
	tensor out of the tone-mapped TAA output, the reprojected colour and
	derivative history, a motion feature and the network's own feedback tensor;
	the CPU runs a single inference on the NPU; and nss_reconstruct.comp applies
	the two results -- per-pixel kernel-prediction filter coefficients and a
	temporal blend tensor -- to produce IMG_NSS_OUTPUT at exactly 2x the render
	extent. The GPU->NPU->GPU round trip is a hard pipeline stall.

	Resolution
	----------
	Unlike a spatial upscaler, NSS was exported at one fixed input shape (see
	tools/export_onnx_int8.py in the neural-super-sampling checkout). It cannot
	tile or crop to an arbitrary render extent, so while it is loaded it pins the
	render extent outright and viewsize/DRS have no effect on it -- see
	vkpt_upscaler_get_temporal_extent() and get_render_extent() in main.c.

	The model's output is exactly 2x its input; that ratio is the only one the
	static KPN tap LUT in nss_reconstruct.comp is valid for. vkpt_final_blit()
	stretches the written sub-rect to fill the actual display, the same mechanism
	DRS uses elsewhere.

	NSS is a consumer of temporal AA rather than an alternative to it:
	nss_pack.comp reads TEX_TAA_OUTPUT and wants the jitter-resolved frame, so
	evaluate_taa_settings() folds AA_MODE_NSS into AA_MODE_UPSCALE and then pins
	TAAU to 1:1, leaving the actual upscaling to the network.

	Q2RTX cvars
	-----------
	* flt_nss_enable - whether the NSS model is loaded. Changing it reloads the
	  ONNX Runtime session, which takes a few seconds because the QNN EP
	  finalizes the HTP graph. Normally driven by flt_taa, which carries it as a
	  fourth anti-aliasing mode.
	* flt_upscaler_verbose - raise ONNX Runtime logging to verbose, which is
	  where per-node execution-provider assignment is reported. That assignment
	  is the authoritative signal that the graph really lands on the Hexagon NPU
	  rather than silently falling back to CPU. Read once, when the ONNX Runtime
	  environment is created, so it must be set before the model loads.

	Console commands
	----------------
	* upscaler_dump - dump the pre-inference input tensor's history and colour
	  channels, and the post-inference temporal blend parameters, for the next
	  rendered frame as PNGs under <gamedir>/screenshots/upscaler/.
*/

#include "shared/shared.h"
#include "common/common.h"
#include "vkpt.h"

#include <string.h>
#include <stdlib.h>

cvar_t *cvar_flt_nss_enable = NULL;
cvar_t *cvar_flt_upscaler_verbose = NULL;

extern cvar_t *cvar_flt_fsr_enable; // owned by fsr.c, initialized just before us
extern cvar_t *cvar_flt_taa;        // owned by main.c's UBO_CVAR_LIST, registered just before us

// Brings the model in line with flt_nss_enable. No-op until the ONNX Runtime
// side is up, and cheap when the selection did not actually change.
static void upscaler_reload_model(void);

// flt_taa doubles as the selector for the temporal upscaler: AA_MODE_NSS is a
// fourth entry in the anti-aliasing menu rather than an entry in the upscaling
// one, because NSS reconstructs from frame history like TAA/TAAU do and unlike
// FSR -- it replaces the temporal AA stage rather than competing with a spatial
// upscaler. The menu cvar is a selector; flt_nss_enable stays authoritative.
// FSR is the one genuine conflict, since it also wants to be the thing that
// resolves to display resolution.
static void aa_mode_changed(cvar_t *self)
{
	bool enable = (self->integer == AA_MODE_NSS);

	if (enable)
		Cvar_SetInteger(cvar_flt_fsr_enable, 0, FROM_CODE);

	if (cvar_flt_nss_enable->integer == (enable ? 1 : 0))
		return;

	Cvar_SetInteger(cvar_flt_nss_enable, enable ? 1 : 0, FROM_CODE);
	upscaler_reload_model();
}

void vkpt_upscaler_init_cvars(void)
{
	cvar_flt_nss_enable = Cvar_Get("flt_nss_enable", "0", CVAR_ARCHIVE);
	// Dumps ONNX Runtime's per-node execution-provider assignment at startup,
	// which is how you confirm the model is really running on the NPU.
	cvar_flt_upscaler_verbose = Cvar_Get("flt_upscaler_verbose", "0", 0);

	// flt_taa is registered by the UBO_CVAR_LIST macro in main.c with no flags,
	// so unlike flt_nss_enable it is not CVAR_ARCHIVE and cannot remember the NSS
	// selection by itself. Persistence comes from flt_nss_enable: reflect it back
	// into flt_taa here so the menu comes up on the mode the user left it in.
	cvar_flt_taa->changed = aa_mode_changed;
	if (cvar_flt_nss_enable->integer)
		Cvar_SetInteger(cvar_flt_taa, AA_MODE_NSS, FROM_CODE);
}

#ifdef USE_ORT_QNN_UPSCALER

#include <windows.h>
#include "onnxruntime_c_api.h"
#include "vk_util.h"
#include "stb_image_write.h"

#define NSS_MODEL_PATH "models/nss-temporal-high-int8.onnx"
#define NSS_MODEL_NAME "NSS Temporal (NPU)"

// ONNX Runtime state, shared by everything below and outliving any one session:
// switching models releases the session but keeps the environment and the CPU
// allocator.
struct
{
	const OrtApi  *api;
	OrtEnv        *env;
	OrtMemoryInfo *cpu_memory_info;
	bool           initialized; // env/allocator are up; safe to load a model
} ort;

#define UPSCALER_TIMING_INTERVAL 100 // frames between timing reports

// ========================================================================== //
// NSS temporal path.
//
// Arm's Neural Super Sampling model (see baseq2/models/nss-temporal-high-int8
// .metadata.json) takes one fixed-shape float32 input tensor (12 channels:
// history/colour/motion/feedback/derivative -- see nss_pack.comp) and produces
// two float32 outputs (KPN filter coefficients and a temporal blend tensor --
// see nss_reconstruct.comp), in a single full-frame dispatch per side.
//
// The element type is float32 even though the export is int8: the QAT export
// bakes QuantizeLinear/DequantizeLinear inside the graph rather than quantizing
// at the graph boundary, so the tensors crossing the API are float.
#define NSS_INPUT_NAME         "_PreprocessTensor"
#define NSS_KPN_NAME           "_KpnCoefficients"
#define NSS_TEMPORAL_NAME      "_TemporalTensor"
#define NSS_NUM_DIMS           4
#define NSS_INPUT_CHANNELS     12
#define NSS_KPN_CHANNELS       36 // "high" preset, kpn_size=6x6
#define NSS_TEMPORAL_CHANNELS  4

struct
{
	OrtSession *session;
	bool        model_loaded;

	int64_t input_dims[NSS_NUM_DIMS];    // [1,12,H,W]
	int64_t kpn_dims[NSS_NUM_DIMS];      // [1,36,H/4,W/4]
	int64_t temporal_dims[NSS_NUM_DIMS]; // [1,4,H,W]
	size_t  input_byte_size, kpn_byte_size, temporal_byte_size; // float32 elements
	uint32_t width, height; // input_dims[3]/[2] -- the one render extent this model accepts

	bool                  pipelines_ready;
	VkPipeline            pipeline_pack;
	VkPipeline            pipeline_reconstruct;
	VkPipelineLayout      pipeline_layout;
	VkDescriptorSetLayout desc_set_layout;
	VkDescriptorPool      desc_pool;
	VkDescriptorSet       desc_set;

	// Host-visible staging for the GPU <-> NPU round trip. Adreno is a
	// unified-memory part, so the compute shaders read/write these directly and
	// the CPU maps them persistently -- no separate device-local copy. Fixed-size
	// and allocated once per model load: NSS's shape does not follow viewsize/DRS.
	BufferResource_t buf_input, buf_kpn, buf_temporal;
	void            *input_mapped, *kpn_mapped, *temporal_mapped;

	// Set by nss_do(), consumed by nss_run_inference().
	bool             packed_this_frame;
	// Set once inference has actually filled the output tensors, so the blit can
	// tell a real result from a frame where the pass bailed.
	bool             tensor_valid;

	// One-shot flag set by the upscaler_dump console command, consumed and
	// cleared by nss_run_inference().
	bool             dump_requested;

	// Rolling cost of the NPU round trip. Accumulated over many frames because
	// Sys_Milliseconds() is far too coarse to time a single frame's inference.
	unsigned         inference_ms_accum;
	unsigned         inference_frames;
} nss;

// Implementations sit further down; forward-declared here so the entry points
// above them can call them.
static bool nss_load_model(void);
static void nss_unload_model(void);
static VkResult nss_create_pipelines(void);
static VkResult nss_destroy_pipelines(void);
static VkResult nss_do(VkCommandBuffer cmd_buf);
static VkResult nss_run_inference(void);
static void     nss_record_reconstruct(VkCommandBuffer cmd_buf);
static VkResult nss_final_blit(VkCommandBuffer cmd_buf, bool warp);

static bool nss_is_active(void)
{
	return cvar_flt_nss_enable->integer != 0 && nss.model_loaded && nss.pipelines_ready;
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

	Com_EPrintf("upscaler: %s failed: %s\n", what, ort.api->GetErrorMessage(status));
	ort.api->ReleaseStatus(status);
	return false;
}

// Queries the shape, element type and byte size of one input/output tensor of
// `session`. Generic in the element type so the same routine covers a model
// whose tensors cross the API as float and one that quantizes at the boundary.
static bool query_tensor_shape(OrtSession *session, bool is_input, size_t index, const char *expected_name,
	ONNXTensorElementDataType expected_elem_type, int64_t *dims_out, size_t num_dims_expected, size_t *byte_size_out)
{
	OrtAllocator *allocator;
	if (!ort_ok(ort.api->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions"))
		return false;

	char *name = NULL;
	OrtTypeInfo *type_info = NULL;
	bool ok = false;

	OrtStatus *status = is_input
		? ort.api->SessionGetInputName(session, index, allocator, &name)
		: ort.api->SessionGetOutputName(session, index, allocator, &name);
	if (!ort_ok(status, "SessionGetInput/OutputName"))
		return false;

	if (strcmp(name, expected_name) != 0) {
		Com_WPrintf("upscaler: model %s %zu is named '%s', expected '%s'\n",
			is_input ? "input" : "output", index, name, expected_name);
	}

	status = is_input
		? ort.api->SessionGetInputTypeInfo(session, index, &type_info)
		: ort.api->SessionGetOutputTypeInfo(session, index, &type_info);
	if (!ort_ok(status, "SessionGetInput/OutputTypeInfo"))
		goto done;

	{
		const OrtTensorTypeAndShapeInfo *tensor_info;
		if (!ort_ok(ort.api->CastTypeInfoToTensorInfo(type_info, &tensor_info), "CastTypeInfoToTensorInfo"))
			goto done;

		size_t num_dims;
		if (!ort_ok(ort.api->GetDimensionsCount(tensor_info, &num_dims), "GetDimensionsCount"))
			goto done;

		if (num_dims != num_dims_expected) {
			Com_EPrintf("upscaler: expected a %zu-D tensor, model %s %zu has %zu dims\n",
				num_dims_expected, is_input ? "input" : "output", index, num_dims);
			goto done;
		}

		if (!ort_ok(ort.api->GetDimensions(tensor_info, dims_out, num_dims), "GetDimensions"))
			goto done;

		ONNXTensorElementDataType elem_type;
		if (!ort_ok(ort.api->GetTensorElementType(tensor_info, &elem_type), "GetTensorElementType"))
			goto done;

		if (elem_type != expected_elem_type) {
			Com_EPrintf("upscaler: expected element type %d, model %s %zu has element type %d\n",
				(int)expected_elem_type, is_input ? "input" : "output", index, (int)elem_type);
			goto done;
		}

		size_t element_count = 1;
		for (size_t i = 0; i < num_dims; i++)
			element_count *= (size_t)dims_out[i];
		*byte_size_out = element_count * (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint8_t));

		Com_Printf("upscaler: model %s '%s' shape [%lld,%lld,%lld,%lld]\n",
			is_input ? "input" : "output", name,
			(long long)dims_out[0], (long long)dims_out[1], (long long)dims_out[2], (long long)dims_out[3]);

		ok = true;
	}

done:
	if (type_info)
		ort.api->ReleaseTypeInfo(type_info);
	if (name)
		ort.api->AllocatorFree(allocator, name);
	return ok;
}

// Writes one 3-channel group out of an NCHW float32 tensor as a PNG, for
// visually checking nss_pack.comp's input tensor. Values are assumed already
// in [0,1] display-referred range (true for the history/colour channels --
// see nss_pack.comp's header), so this is just a float->byte round. The KPN
// output is not dumped this way: its channels are per-tap filter weights, not
// an RGB image.
static void nss_dump_channels(const char *label, const float *tensor, int channel_base, uint32_t width, uint32_t height)
{
	uint8_t *interleaved = Z_Malloc((size_t)width * height * 3);
	size_t plane = (size_t)width * height;

	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			size_t src = (size_t)y * width + x;
			size_t dst = src * 3;
			for (int c = 0; c < 3; c++) {
				float v = tensor[(channel_base + c) * plane + src];
				v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
				interleaved[dst + c] = (uint8_t)(v * 255.0f + 0.5f);
			}
		}
	}

	char path[MAX_OSPATH];
	if (Q_snprintf(path, sizeof(path), "%s/screenshots/upscaler/nss_%s_%" PRIu64 ".png",
			fs_gamedir, label, qvk.frame_counter) >= sizeof(path))
	{
		Com_EPrintf("upscaler: NSS dump path too long\n");
		goto done;
	}

	if (FS_CreatePath(path) < 0) {
		Com_EPrintf("upscaler: failed to create directory for '%s'\n", path);
		goto done;
	}

	if (!stbi_write_png(path, width, height, 3, interleaved, width * 3))
		Com_EPrintf("upscaler: failed to write '%s'\n", path);
	else
		Com_Printf("upscaler: wrote %s\n", path);

done:
	Z_Free(interleaved);
}

static void Upscaler_Dump_f(void)
{
	nss.dump_requested = true;
	Com_Printf("upscaler: will dump pre/post-inference tensors for the next frame\n");
}

static const cmdreg_t upscaler_cmds[] = {
	{ "upscaler_dump", &Upscaler_Dump_f, NULL },
	{ NULL, NULL, NULL }
};

// Session options + the QNN HTP execution provider + CreateSession, given a
// model file relative to the game dir.
static bool create_qnn_session(const char *model_file, const char *display_name, OrtSession **out_session)
{
	OrtSessionOptions *session_options = NULL;
	if (!ort_ok(ort.api->CreateSessionOptions(&session_options), "CreateSessionOptions"))
		return false;

	{
		const char *provider_keys[]   = { "backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode" };
		const char *provider_values[] = { "QnnHtp.dll", "high_performance", "3" };
		OrtStatus *ep_status = ort.api->SessionOptionsAppendExecutionProvider(session_options, "QNN",
			provider_keys, provider_values, LENGTH(provider_keys));
		if (!ort_ok(ep_status, "SessionOptionsAppendExecutionProvider(QNN)")) {
			Com_Printf("upscaler: QNN execution provider unavailable, NPU upscaler disabled\n");
			ort.api->ReleaseSessionOptions(session_options);
			return false;
		}
	}

	char model_path[MAX_OSPATH];
	if (Q_concat(model_path, sizeof(model_path), fs_gamedir, PATH_SEP_STRING, model_file) >= sizeof(model_path)) {
		Com_EPrintf("upscaler: model path too long\n");
		ort.api->ReleaseSessionOptions(session_options);
		return false;
	}

	Com_Printf("upscaler: loading %s (%s), this takes a moment...\n", model_file, display_name);

	// CreateSession takes a wide path. On failure wmodel_path would be left
	// uninitialized stack, so don't hand it over.
	WCHAR wmodel_path[MAX_OSPATH];
	if (!MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wmodel_path, MAX_OSPATH)) {
		Com_EPrintf("upscaler: could not convert model path '%s' to UTF-16 (error %lu)\n",
			model_path, GetLastError());
		ort.api->ReleaseSessionOptions(session_options);
		return false;
	}

	OrtStatus *session_status = ort.api->CreateSession(ort.env, wmodel_path, session_options, out_session);
	ort.api->ReleaseSessionOptions(session_options);

	if (!ort_ok(session_status, "CreateSession")) {
		Com_Printf("upscaler: could not load %s, %s unavailable "
			"(the model ships in baseq2/models; restore with 'git checkout -- baseq2/models')\n",
			model_path, display_name);
		return false;
	}
	return true;
}

// Brings the model and its pipelines in line with the selector.
//
// The pipelines are normally built during VKPT_INIT_RELOAD_SHADER, which has
// already run by the time the user picks the mode from the menu, and
// nss_create_pipelines() no-ops when no model is loaded. So a model loaded this
// late has to build them here; otherwise pipelines_ready stays false and
// nss_is_active() -- which also decides the render extent -- would not become
// true until something else happened to reload shaders.
static void nss_load_model_and_pipelines(bool enable)
{
	if (!ort.initialized)
		return;

	if (!enable) {
		if (!nss.model_loaded)
			return;
		if (qvk.device)
			vkDeviceWaitIdle(qvk.device);
		nss_destroy_pipelines();
		nss_unload_model();
		return;
	}

	if (!nss.model_loaded) {
		if (!nss_load_model())
			return;

		Com_Printf("upscaler: %s loaded, %ux%u fixed temporal render extent; "
			"set flt_upscaler_verbose 1 to verify NPU dispatch\n",
			NSS_MODEL_NAME, nss.width, nss.height);
	}

	// Keyed on pipelines_ready rather than on "did the model just load", so a
	// model that is still loaded but whose pipelines were torn down (a shader
	// reload between load and here) recovers instead of being stuck inactive.
	if (qvk.device && !nss.pipelines_ready) {
		nss_destroy_pipelines();
		nss_create_pipelines();
	}
}

// Reached from the menu via aa_mode_changed(), and from setting flt_nss_enable
// straight from the console.
static void upscaler_reload_model(void)
{
	nss_load_model_and_pipelines(cvar_flt_nss_enable->integer != 0);
}

static void nss_enable_changed(cvar_t *self)
{
	nss_load_model_and_pipelines(self->integer != 0);
}

VkResult vkpt_upscaler_initialize(void)
{
	memset(&ort, 0, sizeof(ort));
	memset(&nss, 0, sizeof(nss));

	ort.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (!ort.api) {
		Com_EPrintf("upscaler: failed to get ONNX Runtime API (expected ABI version %d)\n", ORT_API_VERSION);
		return VK_SUCCESS;
	}

	// Verbose severity is where ONNX Runtime reports per-node execution-provider
	// assignment, which is the authoritative signal that inference dispatches to
	// the Hexagon NPU rather than silently falling back to CPU. It's far too
	// chatty for normal play (it spams the console every frame), so it's opt-in.
	OrtLoggingLevel log_level = (cvar_flt_upscaler_verbose && cvar_flt_upscaler_verbose->integer)
		? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING;

	if (!ort_ok(ort.api->CreateEnvWithCustomLogger(upscaler_ort_log, NULL, log_level,
			"q2rtx_upscaler", &ort.env), "CreateEnvWithCustomLogger"))
		return VK_SUCCESS;

	if (!ort_ok(ort.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort.cpu_memory_info), "CreateCpuMemoryInfo")) {
		ort.api->ReleaseEnv(ort.env);
		ort.env = NULL;
		return VK_SUCCESS;
	}

	Cmd_Register(upscaler_cmds);

	ort.initialized = true;

	// Toggling the mode from the menu or the console reloads the session;
	// init_cvars runs long before we exist, so the callback is only hooked up
	// here, and nss_load_model_and_pipelines() no-ops until initialized is set.
	cvar_flt_nss_enable->changed = nss_enable_changed;
	nss_load_model_and_pipelines(cvar_flt_nss_enable->integer != 0);

	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy(void)
{
	nss_unload_model();

	ort.initialized = false;

	// Both callbacks point at functions in this file, so they have to go before
	// it stops being able to service them -- otherwise setting either cvar after
	// teardown re-enters a torn-down module.
	cvar_flt_nss_enable->changed = NULL;
	cvar_flt_taa->changed = NULL;

	Cmd_Deregister(upscaler_cmds);

	if (ort.cpu_memory_info) {
		ort.api->ReleaseMemoryInfo(ort.cpu_memory_info);
		ort.cpu_memory_info = NULL;
	}
	if (ort.env) {
		ort.api->ReleaseEnv(ort.env);
		ort.env = NULL;
	}

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

VkResult vkpt_upscaler_create_pipelines(void)
{
	return nss_create_pipelines();
}

VkResult vkpt_upscaler_destroy_pipelines(void)
{
	return nss_destroy_pipelines();
}

bool vkpt_upscaler_is_enabled(void)
{
	return nss_is_active();
}

bool vkpt_upscaler_get_temporal_extent(VkExtent2D *out)
{
	if (!nss_is_active())
		return false;
	out->width  = nss.width;
	out->height = nss.height;
	return true;
}

VkResult vkpt_upscaler_do(VkCommandBuffer cmd_buf)
{
	return nss_do(cmd_buf);
}

VkResult vkpt_upscaler_run_inference(void)
{
	return nss_run_inference();
}

VkResult vkpt_upscaler_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	return nss_final_blit(cmd_buf, warp);
}

// ========================================================================== //

static bool nss_load_model(void)
{
	if (!create_qnn_session(NSS_MODEL_PATH, NSS_MODEL_NAME, &nss.session))
		return false;

	if (!query_tensor_shape(nss.session, true, 0, NSS_INPUT_NAME, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			nss.input_dims, NSS_NUM_DIMS, &nss.input_byte_size) ||
		!query_tensor_shape(nss.session, false, 0, NSS_KPN_NAME, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			nss.kpn_dims, NSS_NUM_DIMS, &nss.kpn_byte_size) ||
		!query_tensor_shape(nss.session, false, 1, NSS_TEMPORAL_NAME, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			nss.temporal_dims, NSS_NUM_DIMS, &nss.temporal_byte_size))
	{
		nss_unload_model();
		return false;
	}

	// Matches the fixed "high" NSS contract this integration targets -- see
	// baseq2/models/nss-temporal-high-int8.metadata.json. A model that does not
	// match (e.g. the mid/low variant, with 16 KPN channels and a different
	// fixed shape) is rejected rather than partially supported.
	bool shape_ok =
		nss.input_dims[0] == 1 && nss.input_dims[1] == NSS_INPUT_CHANNELS &&
		nss.kpn_dims[0] == 1 && nss.kpn_dims[1] == NSS_KPN_CHANNELS &&
		nss.temporal_dims[0] == 1 && nss.temporal_dims[1] == NSS_TEMPORAL_CHANNELS &&
		nss.input_dims[2] > 0 && nss.input_dims[3] > 0 &&
		nss.input_dims[2] % 8 == 0 && nss.input_dims[3] % 8 == 0 &&
		nss.kpn_dims[2] == nss.input_dims[2] / 4 && nss.kpn_dims[3] == nss.input_dims[3] / 4 &&
		nss.temporal_dims[2] == nss.input_dims[2] && nss.temporal_dims[3] == nss.input_dims[3];

	if (!shape_ok) {
		Com_EPrintf("upscaler: %s does not match the NSS 'high' contract "
			"(12ch float32 in; 36ch KPN out at 1/4 res; 4ch temporal out at full res; "
			"dims a multiple of 8); not loading\n", NSS_MODEL_PATH);
		nss_unload_model();
		return false;
	}

	nss.width  = (uint32_t)nss.input_dims[3];
	nss.height = (uint32_t)nss.input_dims[2];
	nss.model_loaded = true;
	return true;
}

static void nss_destroy_tensor_buffers(void)
{
	if (nss.input_mapped) {
		buffer_unmap(&nss.buf_input);
		nss.input_mapped = NULL;
	}
	if (nss.kpn_mapped) {
		buffer_unmap(&nss.buf_kpn);
		nss.kpn_mapped = NULL;
	}
	if (nss.temporal_mapped) {
		buffer_unmap(&nss.buf_temporal);
		nss.temporal_mapped = NULL;
	}

	buffer_destroy(&nss.buf_input);
	buffer_destroy(&nss.buf_kpn);
	buffer_destroy(&nss.buf_temporal);
}

static void nss_unload_model(void)
{
	if (nss.session) {
		ort.api->ReleaseSession(nss.session);
		nss.session = NULL;
	}

	// The staging buffers are sized from the model's tensor shapes, so they
	// cannot outlive it. The dispatches referencing them may still be in flight,
	// hence the wait.
	if (qvk.device)
		vkDeviceWaitIdle(qvk.device);
	nss_destroy_tensor_buffers();

	nss.model_loaded = false;
	nss.width = 0;
	nss.height = 0;
}

// (Re)allocates the fixed-size tensor staging buffers and points the
// descriptor set at them. There is no per-frame resize case: NSS's shape does
// not follow viewsize/DRS (see get_render_extent() in main.c), so this only
// ever needs to run once per model load, from nss_create_pipelines().
static bool nss_ensure_tensor_buffers(void)
{
	if (nss.input_mapped)
		return true;

	const VkMemoryPropertyFlags host_props =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	if (buffer_create(&nss.buf_input, nss.input_byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS ||
		buffer_create(&nss.buf_kpn, nss.kpn_byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS ||
		buffer_create(&nss.buf_temporal, nss.temporal_byte_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS)
	{
		Com_EPrintf("upscaler: failed to allocate NSS tensor buffers\n");
		nss_destroy_tensor_buffers();
		return false;
	}

	buffer_attach_name(&nss.buf_input, "nss input tensor");
	buffer_attach_name(&nss.buf_kpn, "nss kpn tensor");
	buffer_attach_name(&nss.buf_temporal, "nss temporal tensor");

	nss.input_mapped    = buffer_map(&nss.buf_input);
	nss.kpn_mapped      = buffer_map(&nss.buf_kpn);
	nss.temporal_mapped = buffer_map(&nss.buf_temporal);

	if (!nss.input_mapped || !nss.kpn_mapped || !nss.temporal_mapped) {
		Com_EPrintf("upscaler: failed to map NSS tensor buffers\n");
		nss_destroy_tensor_buffers();
		return false;
	}

	// nss_pack.comp reads this as feedback history before the first inference
	// has ever run; zero is the network's own "no history yet" state (the same
	// state disocclusion resets to every frame after), not an arbitrary choice.
	memset(nss.temporal_mapped, 0, nss.temporal_byte_size);

	VkDescriptorBufferInfo buffer_info[] = {
		{ .buffer = nss.buf_input.buffer,    .offset = 0, .range = nss.input_byte_size },
		{ .buffer = nss.buf_kpn.buffer,      .offset = 0, .range = nss.kpn_byte_size },
		{ .buffer = nss.buf_temporal.buffer, .offset = 0, .range = nss.temporal_byte_size },
	};
	VkWriteDescriptorSet writes[3];
	for (int i = 0; i < 3; i++) {
		writes[i] = (VkWriteDescriptorSet){
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = nss.desc_set,
			.dstBinding      = (uint32_t)i,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo     = &buffer_info[i],
		};
	}
	vkUpdateDescriptorSets(qvk.device, LENGTH(writes), writes, 0, NULL);

	Com_Printf("upscaler: NSS tensor buffers ready (%.1f MB staging)\n",
		(double)(nss.input_byte_size + nss.kpn_byte_size + nss.temporal_byte_size) / (1024.0 * 1024.0));

	return true;
}

static VkResult nss_create_pipelines(void)
{
	if (!nss.model_loaded)
		return VK_SUCCESS;

	// binding 0: input tensor, written by nss_pack.comp.
	// binding 1: KPN coefficients, read by nss_reconstruct.comp.
	// binding 2: temporal tensor -- ONNX Runtime's Run() output, read back by
	//   both nss_pack.comp (as reprojected feedback history) and
	//   nss_reconstruct.comp (for the temporal blend params); see
	//   nss_pack.comp's header comment for why this needs no GPU-image ping-pong.
	VkDescriptorSetLayoutBinding bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
	};

	VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = LENGTH(bindings),
		.pBindings    = bindings,
	};
	_VK(vkCreateDescriptorSetLayout(qvk.device, &layout_info, NULL, &nss.desc_set_layout));
	ATTACH_LABEL_VARIABLE(nss.desc_set_layout, DESCRIPTOR_SET_LAYOUT);

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
	_VK(vkCreateDescriptorPool(qvk.device, &pool_info, NULL, &nss.desc_pool));

	VkDescriptorSetAllocateInfo alloc_info = {
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool     = nss.desc_pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &nss.desc_set_layout,
	};
	_VK(vkAllocateDescriptorSets(qvk.device, &alloc_info, &nss.desc_set));

	VkDescriptorSetLayout desc_set_layouts[] = {
		qvk.desc_set_layout_ubo,
		qvk.desc_set_layout_textures,
		nss.desc_set_layout,
	};

	// Both nss_pack.comp and nss_reconstruct.comp read the render/display
	// extent straight off global_ubo, so this needs no push constants.
	CREATE_PIPELINE_LAYOUT(qvk.device, &nss.pipeline_layout,
		.setLayoutCount = LENGTH(desc_set_layouts),
		.pSetLayouts    = desc_set_layouts,
	);
	ATTACH_LABEL_VARIABLE(nss.pipeline_layout, PIPELINE_LAYOUT);

	VkComputePipelineCreateInfo pipeline_info[] = {
		{
			.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage  = SHADER_STAGE(QVK_MOD_NSS_PACK_COMP, VK_SHADER_STAGE_COMPUTE_BIT),
			.layout = nss.pipeline_layout,
		},
		{
			.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage  = SHADER_STAGE(QVK_MOD_NSS_RECONSTRUCT_COMP, VK_SHADER_STAGE_COMPUTE_BIT),
			.layout = nss.pipeline_layout,
		},
	};

	VkPipeline pipelines[LENGTH(pipeline_info)];
	_VK(vkCreateComputePipelines(qvk.device, 0, LENGTH(pipeline_info), pipeline_info, 0, pipelines));

	nss.pipeline_pack        = pipelines[0];
	nss.pipeline_reconstruct = pipelines[1];

	if (!nss_ensure_tensor_buffers())
		return VK_SUCCESS; // pipelines_ready stays false; destroy_pipelines cleans up what was created above

	nss.pipelines_ready = true;

	return VK_SUCCESS;
}

static VkResult nss_destroy_pipelines(void)
{
	nss_destroy_tensor_buffers();

	nss.pipelines_ready = false;

	if (nss.pipeline_pack) {
		vkDestroyPipeline(qvk.device, nss.pipeline_pack, NULL);
		nss.pipeline_pack = VK_NULL_HANDLE;
	}
	if (nss.pipeline_reconstruct) {
		vkDestroyPipeline(qvk.device, nss.pipeline_reconstruct, NULL);
		nss.pipeline_reconstruct = VK_NULL_HANDLE;
	}
	if (nss.pipeline_layout) {
		vkDestroyPipelineLayout(qvk.device, nss.pipeline_layout, NULL);
		nss.pipeline_layout = VK_NULL_HANDLE;
	}
	if (nss.desc_pool) {
		vkDestroyDescriptorPool(qvk.device, nss.desc_pool, NULL);
		nss.desc_pool = VK_NULL_HANDLE;
		nss.desc_set = VK_NULL_HANDLE;
	}
	if (nss.desc_set_layout) {
		vkDestroyDescriptorSetLayout(qvk.device, nss.desc_set_layout, NULL);
		nss.desc_set_layout = VK_NULL_HANDLE;
	}

	return VK_SUCCESS;
}

static void nss_bind_pipeline(VkCommandBuffer cmd_buf, VkPipeline pipeline)
{
	VkDescriptorSet desc_sets[] = {
		qvk.desc_set_ubo,
		qvk_get_current_desc_set_textures(),
		nss.desc_set,
	};

	vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
		nss.pipeline_layout, 0, LENGTH(desc_sets), desc_sets, 0, NULL);
}

// Single full-frame dispatch: NSS's fixed shape already covers the whole render
// extent in one shot.
static VkResult nss_do(VkCommandBuffer cmd_buf)
{
	nss.packed_this_frame = false;
	nss.tensor_valid = false;

	if (!nss.pipelines_ready)
		return VK_SUCCESS;

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_PACK);

	nss_bind_pipeline(cmd_buf, nss.pipeline_pack);
	vkCmdDispatch(cmd_buf, (nss.width + 7) / 8, (nss.height + 7) / 8, 1);

	// Make the shader writes visible to the host read in nss_run_inference().
	BUFFER_BARRIER(cmd_buf,
		.buffer        = nss.buf_input.buffer,
		.offset        = 0,
		.size          = VK_WHOLE_SIZE,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	);

	// nss_pack.comp also writes IMG_NSS_LUMA_DERIV_A directly (not through the
	// tensor buffer); make that visible before it is next read as history.
	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_NSS_LUMA_DERIV_A]);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_PACK);
	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);

	nss.packed_this_frame = true;

	return VK_SUCCESS;
}

// The NPU runs on the CPU's side of the fence: must run after the command
// buffer holding nss_do()'s pack dispatch is submitted, and before
// nss_final_blit(). One inference for the whole frame.
static VkResult nss_run_inference(void)
{
	if (!nss.packed_this_frame)
		return VK_SUCCESS;

	nss.packed_this_frame = false;

	bool dump = nss.dump_requested;
	nss.dump_requested = false;

	vkQueueWaitIdle(qvk.queue_graphics);

	unsigned time_begin = Sys_Milliseconds();

	if (dump) {
		const float *in = (const float *)nss.input_mapped;
		nss_dump_channels("pre_history", in, 0, nss.width, nss.height);
		nss_dump_channels("pre_colour", in, 3, nss.width, nss.height);
	}

	OrtValue *input_value = NULL, *kpn_value = NULL, *temporal_value = NULL;
	bool ok = false;

	if (!ort_ok(ort.api->CreateTensorWithDataAsOrtValue(ort.cpu_memory_info, nss.input_mapped,
			nss.input_byte_size, nss.input_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_value), "CreateTensorWithDataAsOrtValue(nss input)"))
		goto done;

	if (!ort_ok(ort.api->CreateTensorWithDataAsOrtValue(ort.cpu_memory_info, nss.kpn_mapped,
			nss.kpn_byte_size, nss.kpn_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &kpn_value), "CreateTensorWithDataAsOrtValue(nss kpn)"))
		goto done;

	if (!ort_ok(ort.api->CreateTensorWithDataAsOrtValue(ort.cpu_memory_info, nss.temporal_mapped,
			nss.temporal_byte_size, nss.temporal_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &temporal_value), "CreateTensorWithDataAsOrtValue(nss temporal)"))
		goto done;

	{
		const char *input_names[]  = { NSS_INPUT_NAME };
		const char *output_names[] = { NSS_KPN_NAME, NSS_TEMPORAL_NAME };
		const OrtValue *inputs[]   = { input_value };
		OrtValue *outputs[]        = { kpn_value, temporal_value };

		OrtStatus *status = ort.api->Run(nss.session, NULL,
			input_names, inputs, 1, output_names, 2, outputs);

		if (status) {
			Com_EPrintf("upscaler: NSS inference failed: %s\n", ort.api->GetErrorMessage(status));
			ort.api->ReleaseStatus(status);
		} else {
			ok = true;
			if (dump) {
				// Not an RGB image -- channels 0/1/2 are theta/alpha/gamma blend
				// params (see nss_reconstruct.comp's SampleTemporalParams), dumped
				// here only as a rough "did this change frame to frame" sanity check.
				nss_dump_channels("post_temporal_params", (const float *)nss.temporal_mapped, 0, nss.width, nss.height);
			}
		}
	}

done:
	if (input_value)
		ort.api->ReleaseValue(input_value);
	if (kpn_value)
		ort.api->ReleaseValue(kpn_value);
	if (temporal_value)
		ort.api->ReleaseValue(temporal_value);

	nss.tensor_valid = ok;

	nss.inference_ms_accum += Sys_Milliseconds() - time_begin;
	nss.inference_frames++;
	if (nss.inference_frames >= UPSCALER_TIMING_INTERVAL) {
		Com_Printf("upscaler: NSS %.2f ms/frame\n",
			(double)nss.inference_ms_accum / nss.inference_frames);
		nss.inference_ms_accum = 0;
		nss.inference_frames = 0;
	}

	return VK_SUCCESS;
}

// Records the reconstruct pass into VKPT_IMG_NSS_OUTPUT.
static void nss_record_reconstruct(VkCommandBuffer cmd_buf)
{
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);

	nss_bind_pipeline(cmd_buf, nss.pipeline_reconstruct);

	// Dispatches at the model's true output extent (render extent at an exact
	// 2x scale -- the only ratio the static KPN tap LUT is valid for, see
	// nss_reconstruct.comp's header) and writes that corner sub-rect in one pass.
	VkExtent2D out = { nss.width * 2, nss.height * 2 };
	vkCmdDispatch(cmd_buf, (out.width + 7) / 8, (out.height + 7) / 8, 1);

	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_NSS_OUTPUT]);
	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_NSS_HISTORY_COLOR_A]);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);
}

static VkResult nss_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	// A failed inference, or a frame where the pass never ran, leaves nothing
	// valid to reconstruct from.
	if (!nss.tensor_valid) {
		bool needs_filter = qvk.extent_taa_output.width  != qvk.extent_unscaled.width
		                 || qvk.extent_taa_output.height != qvk.extent_unscaled.height;
		return vkpt_final_blit(cmd_buf, VKPT_IMG_TAA_OUTPUT, qvk.extent_taa_output,
			needs_filter, warp);
	}

	nss_record_reconstruct(cmd_buf);

	// vkpt_final_blit() stretches the written sub-rect to fill the actual
	// display, same mechanism DRS uses elsewhere.
	VkExtent2D out = { nss.width * 2, nss.height * 2 };
	return vkpt_final_blit(cmd_buf, VKPT_IMG_NSS_OUTPUT, out, false, warp);
}

#else // !USE_ORT_QNN_UPSCALER

// No ONNX Runtime in this build, so there is nothing to load. flt_nss_enable and
// the flt_taa hook still exist above, so a config carrying the NSS selection
// round-trips instead of being silently rewritten; the mode just never becomes
// available.
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

// Load-bearing: this returning false is what makes get_render_extent() in
// main.c fall through to its normal viewsize/DRS-derived size.
bool vkpt_upscaler_get_temporal_extent(VkExtent2D *out)
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
