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

	Per frame: upscaler_pack.comp resamples the tone-mapped TAA output into a
	grid of uint8 NCHW input tensors, the CPU runs one inference per tile on the
	NPU, and upscaler_unpack.comp writes the results into IMG_UPSCALE_OUTPUT for
	the final blit. The GPU->NPU->GPU round trip is a hard pipeline stall, which
	is why the tile count has to stay low to be playable.

	The models have static tensor shapes -- every one shipped here is
	1x3x128x128 uint8 in, 1x3x512x512 out -- so the frame is tiled rather than
	fed in whole. The geometry is not hardcoded: it is queried
	off whichever model is loaded and pushed to the shaders, so swapping in a
	model with different fixed shapes needs no code change.

	Resolution
	----------
	The model's scale factor is a property of the model, not a resolution policy.
	viewsize/DRS choose the render extent exactly as they do for every other path;
	the tile grid is then sized to cover that, and the model multiplies it by its
	fixed factor. Whatever that overshoots the display by, upscaler_unpack.comp
	removes on the way out with an area-weighted box filter.

	So the downsample ratio is viewsize * scale / 100. With a 4x model, viewsize 25
	lands on the display exactly and the filter collapses to a 1:1 readback -- the
	cheap upscale-for-performance case. Above that the extra resolution is real
	supersampling, and at viewsize 100 the path tracer runs at native resolution
	and the frame is 4x-downsampled: high quality, and far too slow for gameplay,
	since the tile count and therefore the number of serial NPU inferences grows
	with the square of viewsize. flt_upscaler_max_tiles is the backstop on that.

	None of the above applies to NSS, whose input shape is fixed: it pins the render
	extent outright and ignores viewsize/DRS.

	Chaining
	--------
	The temporal and spatial slots are independent and may both be on, in which
	case they run in series: NSS reconstructs the frame at 2x, and the spatial
	model then upscales *that* rather than the TAA output. The stage order is
	fixed -- NSS emits plain colour, which is all the spatial pack consumes,
	whereas the reverse would desynchronise the colour from the motion vectors and
	depth NSS reprojects with.

	The cost is a second GPU -> NPU -> GPU round trip: the spatial pack cannot be
	recorded until NSS's reconstruction exists, so vkpt_upscaler_run_inference()
	submits a command buffer of its own in between and stalls on it. The spatial
	stage also sees an input already at 2x the render extent, which squares up its
	tile count -- roughly 136 serial inferences per frame at 1080p. Useful for
	evaluating the two together; not a playable configuration.

	Q2RTX cvars
	-----------
	* flt_upscaler_enable - which spatial model to run: 0 = disabled,
	  1 = QuickSRNetSmall, 2 = QuickSRNetLarge, 3 = QuickSRNetLarge fine-tuned on
	  Quake II RTX frames (see upscaler_models[]). Changing it reloads the ONNX
	  Runtime session, which takes a few seconds because the QNN EP finalizes the
	  HTP graph. Normally driven by the flt_upscaling menu cvar.
	* flt_nss_enable - whether the NSS temporal model is loaded. A separate slot
	  from flt_upscaler_enable precisely so the two can be up at once. Normally
	  driven by flt_taa, which carries it as a fourth anti-aliasing mode.
	* flt_upscaler_max_tiles - refuse to run a frame needing more than this many
	  tiles, so an over-ambitious viewsize degrades to the non-upscaled blit
	  instead of stalling for seconds on a huge host-visible allocation.
	* flt_upscaler_verbose - raise ONNX Runtime logging to verbose, which is
	  where per-node execution-provider assignment is reported.

	Console commands
	----------------
	* upscaler_npu_test [seconds] - run repeated inference for the given
	  duration (default 3s) so NPU dispatch/utilization can be confirmed via
	  Windows Task Manager > Performance > NPU while it runs, and via the
	  verbose ONNX Runtime log lines this prints to the console.
	* upscaler_dump - dump the pre- and post-upscale tensor for every tile of
	  the next rendered frame as PNGs under <gamedir>/screenshots/upscaler/,
	  for visually inspecting what goes into and comes out of the model.
*/

#include "shared/shared.h"
#include "common/common.h"
#include "vkpt.h"

#include <string.h>
#include <stdlib.h>

cvar_t *cvar_flt_upscaler_enable = NULL;
cvar_t *cvar_flt_nss_enable = NULL;
cvar_t *cvar_flt_upscaler_verbose = NULL;
cvar_t *cvar_flt_upscaler_max_tiles = NULL;

extern cvar_t *cvar_flt_fsr_enable; // owned by fsr.c, initialized just before us
extern cvar_t *cvar_flt_taa;        // owned by main.c's UBO_CVAR_LIST, registered just before us

cvar_t *cvar_flt_upscaling = NULL;

// Spatial (QuickSRNet-family) models are stateless, tile independently and
// scale by a fixed integer factor from a square uint8 tensor -- see upscaler.c's
// overview comment. Temporal (NSS) models instead run once per frame at a
// fixed shape with cross-frame history; see the "NSS temporal path" section
// below. The two have separate selectors and separate slots, so either, both,
// or neither may be loaded; this tag is what routes each to its own code path
// rather than assuming every model shares QuickSRNet's contract.
typedef enum {
	UPSCALER_KIND_SPATIAL,
	UPSCALER_KIND_TEMPORAL,
} upscaler_kind_t;

// The NPU upscaler models, indexed by flt_upscaler_enable - 1. All ship in the
// repo under baseq2/models. The first three are square fixed-shape uint8 NCHW
// spatial models: the first two are stock w8a8 builds from Qualcomm AI Hub,
// the third a QuickSRNet Large fine-tuned on Quake II RTX frames and
// re-exported with its weights embedded, so it has no .data sidecar. The
// fourth is Arm's Neural Super Sampling temporal model (see
// baseq2/models/nss-temporal-high-int8.metadata.json for its I/O contract).
// Order matters: it defines both the flt_upscaler_enable values and the
// flt_upscaling values below, so append rather than insert.
static const struct {
	const char *name;
	const char *path;
	upscaler_kind_t kind;
} upscaler_models[] = {
	{ "QuickSRNet Small",               "models/quicksrnetsmall-w8a8.onnx",       UPSCALER_KIND_SPATIAL  },
	{ "QuickSRNet Large",               "models/quicksrnetlarge-w8a8.onnx",       UPSCALER_KIND_SPATIAL  },
	{ "QuickSRNet Large (Q2RTX-tuned)", "models/quicksrnetlarge-q2rtx-w8a8.onnx", UPSCALER_KIND_SPATIAL  },
	{ "NSS Temporal (NPU)",             "models/nss-temporal-high-int8.onnx",     UPSCALER_KIND_TEMPORAL },
};

// Menu-facing selector for the mutually exclusive *spatial* upscalers. The
// per-backend cvars stay authoritative so existing configs, scripts and console
// use keep working; this just keeps them from being enabled at the same time.
// The temporal (NSS) model is not in this list -- it is selected from flt_taa,
// alongside the other temporal reconstruction modes; see aa_mode_changed().
enum {
	UPSCALING_MODE_NONE     = 0,
	UPSCALING_MODE_FSR      = 1,
	UPSCALING_MODE_AI_FIRST = 2, // 2 .. 2 + <number of spatial models> - 1
};

// Loads whatever model flt_upscaler_enable now names. No-op until the ONNX
// Runtime side is up, and cheap when the selection did not actually change.
static void upscaler_reload_model(void);

// Table queries on a 1-based model index. Both work without the ONNX Runtime
// side being compiled in, unlike nss_is_active()/spatial_is_active(), which ask
// about the models actually loaded.
static bool upscaler_model_is_temporal(int model)
{
	return model > 0 && model <= (int)LENGTH(upscaler_models)
		&& upscaler_models[model - 1].kind == UPSCALER_KIND_TEMPORAL;
}

// flt_upscaler_enable index of the NSS model, or 0 if this build has none.
// Derived rather than hardcoded so the "append rather than insert" rule on
// upscaler_models[] is the only thing anyone has to remember.
static int upscaler_temporal_model_index(void)
{
	for (int i = 0; i < (int)LENGTH(upscaler_models); i++) {
		if (upscaler_models[i].kind == UPSCALER_KIND_TEMPORAL)
			return i + 1;
	}
	return 0;
}

static void upscaling_mode_changed(cvar_t *self)
{
	// flt_upscaler_enable is a 1-based model index rather than a boolean, so
	// everything that only asks "is the AI upscaler on?" still just tests it
	// against zero, and an archived flt_upscaler_enable 1 still means the
	// original model.
	int model = self->integer - (UPSCALING_MODE_AI_FIRST - 1);
	if (model < 1 || model > (int)LENGTH(upscaler_models))
		model = 0;

	// Temporal models are not reachable from this list -- they have their own
	// slot and their own selector. A config carrying the value NSS used to have
	// lands on "none" rather than selecting a model this list cannot drive.
	if (upscaler_model_is_temporal(model))
		model = 0;

	Cvar_SetInteger(cvar_flt_fsr_enable, self->integer == UPSCALING_MODE_FSR, FROM_CODE);
	Cvar_SetInteger(cvar_flt_upscaler_enable, model, FROM_CODE);

	// Note this deliberately does not touch flt_taa/flt_nss_enable: the temporal
	// and spatial upscalers occupy different stages and can both be on, in which
	// case NSS reconstructs and the spatial model then upscales its output.

	// Cvar_SetInteger(FROM_CODE) deliberately does not run change callbacks
	// (change_string_value() in common/cvar.c), so the swap has to be kicked
	// off from here rather than from flt_upscaler_enable's own callback.
	upscaler_reload_model();
}

// flt_taa doubles as the selector for the temporal upscaler: AA_MODE_NSS is a
// fourth entry in the anti-aliasing menu rather than a fourth entry in the
// upscaling one, because NSS reconstructs from frame history like TAA/TAAU do
// and unlike FSR and QuickSRNet -- it replaces the temporal AA stage rather
// than competing with the spatial upscalers. Same arrangement as
// upscaling_mode_changed(): the menu cvar is a selector, flt_nss_enable stays
// authoritative. FSR is the one genuine conflict, since it also wants to be the
// thing that resolves to display resolution.
static void aa_mode_changed(cvar_t *self)
{
	bool enable = (self->integer == AA_MODE_NSS) && upscaler_temporal_model_index() != 0;

	if (enable)
		Cvar_SetInteger(cvar_flt_fsr_enable, 0, FROM_CODE);

	if (cvar_flt_nss_enable->integer == (enable ? 1 : 0))
		return;

	Cvar_SetInteger(cvar_flt_nss_enable, enable ? 1 : 0, FROM_CODE);
	upscaler_reload_model();
}

void vkpt_upscaler_init_cvars(void)
{
	cvar_flt_upscaler_enable = Cvar_Get("flt_upscaler_enable", "0", CVAR_ARCHIVE);
	// The temporal model gets its own slot rather than sharing the spatial one's
	// index, because the two run at different stages and can be on at once.
	cvar_flt_nss_enable = Cvar_Get("flt_nss_enable", "0", CVAR_ARCHIVE);
	// Dumps ONNX Runtime's per-node execution-provider assignment at startup,
	// which is how you confirm the model is really running on the NPU.
	cvar_flt_upscaler_verbose = Cvar_Get("flt_upscaler_verbose", "0", 0);

	// Backstop on how much work one frame may ask of the NPU. The tile count
	// grows with the square of viewsize, and each tile is a serial inference plus
	// its share of a host-visible staging allocation -- at 1080p a 128 -> 512
	// model needs 12 tiles at viewsize 25 but 135 at 100 and 510 at 200, the last
	// of which would try to map ~400 MB. The default admits the supersampling case
	// and refuses the pathological one; raise it if you have the memory and the
	// patience. Exceeding it drops the pass for that frame rather than degrading
	// the image silently.
	cvar_flt_upscaler_max_tiles = Cvar_Get("flt_upscaler_max_tiles", "256", CVAR_ARCHIVE);

	// upscaling_mode_changed() below overwrites flt_upscaler_enable from the menu
	// cvar, so latch the archived value first. An archived index naming the
	// temporal model is a config from before it had its own slot; migrate it
	// rather than dropping the user's selection on the floor.
	int archived_model = cvar_flt_upscaler_enable->integer;
	bool archived_temporal = upscaler_model_is_temporal(archived_model);

	// Seeded from whatever the backend cvars already say, so a config that set
	// flt_fsr_enable or flt_upscaler_enable directly shows up correctly in the
	// menu, on the right model. A temporal index seeds "none" here -- it belongs
	// to the anti-aliasing selector instead.
	char initial[16];
	if (!archived_temporal && archived_model > 0 &&
		archived_model <= (int)LENGTH(upscaler_models))
	{
		Q_snprintf(initial, sizeof(initial), "%d",
			UPSCALING_MODE_AI_FIRST - 1 + archived_model);
	} else {
		Q_strlcpy(initial, cvar_flt_fsr_enable && cvar_flt_fsr_enable->integer ? "1" : "0", sizeof(initial));
	}

	cvar_flt_upscaling = Cvar_Get("flt_upscaling", initial, CVAR_ARCHIVE);
	cvar_flt_upscaling->changed = upscaling_mode_changed;
	upscaling_mode_changed(cvar_flt_upscaling);

	// flt_taa is registered by the UBO_CVAR_LIST macro in main.c with no flags,
	// so unlike flt_nss_enable it is not CVAR_ARCHIVE and cannot remember the NSS
	// selection by itself. Persistence comes from flt_nss_enable: reflect it back
	// into flt_taa here so the menu comes up on the mode the user left it in.
	cvar_flt_taa->changed = aa_mode_changed;
	if (archived_temporal)
		Cvar_SetInteger(cvar_flt_nss_enable, 1, FROM_CODE);
	if (cvar_flt_nss_enable->integer)
		Cvar_SetInteger(cvar_flt_taa, AA_MODE_NSS, FROM_CODE);
}

#ifdef USE_ORT_QNN_UPSCALER

#include <windows.h>
#include "onnxruntime_c_api.h"
#include "vk_util.h"
#include "stb_image_write.h"

#define UPSCALER_INPUT_NAME "image"
#define UPSCALER_OUTPUT_NAME "upscaled_image"
#define UPSCALER_NUM_DIMS 4
#define UPSCALER_DEFAULT_TEST_SECONDS 3

// Sanity bound on the model's output tile, so a bogus model can't ask for an
// absurd staging allocation before anything else notices.
#define UPSCALER_MAX_TILE 4096

typedef struct {
	uint32_t tiles_x;
	uint32_t tiles_y;
	uint32_t tile_in;  // model input  edge, pixels
	uint32_t tile_out; // model output edge, pixels
	// The rect of the source image the pack pass may read. Pushed rather than
	// taken from the UBO because the source is not always the TAA output: chained
	// behind NSS it is NSS's reconstruction, at a different extent.
	uint32_t src_width;
	uint32_t src_height;
	uint32_t src_is_nss; // 0 = TEX_TAA_OUTPUT, 1 = TEX_NSS_OUTPUT
} upscaler_push_constants_t;

struct
{
	const OrtApi  *api;
	OrtEnv        *env;
	OrtSession    *session;
	OrtMemoryInfo *cpu_memory_info;
	bool           initialized;  // env/allocator are up; safe to load models
	bool           model_loaded;
	int            loaded_model; // 1-based index into upscaler_models, 0 = none

	int64_t        input_dims[UPSCALER_NUM_DIMS];
	int64_t        output_dims[UPSCALER_NUM_DIMS];
	size_t         input_byte_size;  // uint8 tensor, 1 byte/element
	size_t         output_byte_size;
	uint32_t       tile_in;          // input_dims[2..3], validated square
	uint32_t       tile_out;         // output_dims[2..3], validated square
	uint32_t       scale;            // tile_out / tile_in, validated integer

	// Render integration.
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
	// Set once inference has actually filled the output tensor, so the blit can
	// tell a real result from a frame where the pass bailed and the staging
	// buffers may not even exist.
	bool             tensor_valid;

	// One-shot flag set by the upscaler_dump console command, consumed and
	// cleared by vkpt_upscaler_run_inference().
	bool             dump_requested;

	// Rolling cost of the NPU round trip. Accumulated over many frames because
	// Sys_Milliseconds() is far too coarse to time a single frame's inference.
	unsigned         inference_ms_accum;
	unsigned         inference_frames;
} upscaler;

#define UPSCALER_TIMING_INTERVAL 100 // frames between timing reports

// ========================================================================== //
// NSS temporal path.
//
// Arm's Neural Super Sampling model (see baseq2/models/nss-temporal-high-int8
// .metadata.json) has nothing in common with the spatial QuickSRNet contract
// above beyond "an ONNX Runtime session on the QNN EP": one fixed-shape
// float32 input tensor (12 channels: history/colour/motion/feedback/derivative
// -- see nss_pack.comp), two float32 outputs (KPN filter coefficients and a
// temporal blend tensor -- see nss_reconstruct.comp), a single full-frame
// dispatch per side instead of a serial tile loop, and cross-frame history the
// spatial path has no equivalent for. It gets its own struct and dispatch
// functions, reusing only the ONNX Runtime env/API/logging upscaler already
// set up. The public vkpt_upscaler_* entry points branch between this and the
// spatial path by upscaler_models[upscaler.loaded_model - 1].kind.
//
// Unlike the spatial models, NSS was exported at one fixed shape (see
// tools/export_onnx_int8.py in the neural-super-sampling checkout), so its
// render extent cannot follow viewsize/DRS the way the spatial path's does --
// see vkpt_upscaler_get_temporal_extent() and get_render_extent() in main.c.
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

	// Host-visible staging for the GPU <-> NPU round trip, same rationale as
	// upscaler.buf_input/buf_output. Fixed-size and allocated once per model
	// load rather than per-frame: NSS's shape does not follow viewsize/DRS.
	BufferResource_t buf_input, buf_kpn, buf_temporal;
	void            *input_mapped, *kpn_mapped, *temporal_mapped;

	bool             packed_this_frame;
	bool             tensor_valid;

	// One-shot flag set by the upscaler_dump console command, consumed and
	// cleared by nss_run_inference().
	bool             dump_requested;

	unsigned         inference_ms_accum;
	unsigned         inference_frames;
} nss;

// Implementations sit at the end of the file, grouped together as the NSS
// temporal path's counterpart to everything above; forward-declared here so
// the spatial-path functions above them (load_model()/unload_model() and the
// vkpt_upscaler_* entry points) can call them.
static bool nss_load_model(const char *model_file, const char *display_name);
static void nss_unload_model(void);
static VkResult nss_create_pipelines(void);
static VkResult nss_destroy_pipelines(void);
static VkResult nss_do(VkCommandBuffer cmd_buf);
static VkResult nss_run_inference(void);
static void     nss_record_reconstruct(VkCommandBuffer cmd_buf);
static VkResult nss_final_blit(VkCommandBuffer cmd_buf, bool warp);

// The two slots run at different stages and are independently selectable, so
// every vkpt_upscaler_* entry point asks about each rather than branching on a
// single "which kind is loaded" answer. When both are on they chain: NSS
// reconstructs the frame temporally, then the spatial model upscales its output.
static bool nss_is_active(void)
{
	return cvar_flt_nss_enable->integer != 0 && nss.model_loaded && nss.pipelines_ready;
}

static bool spatial_is_active(void)
{
	return cvar_flt_upscaler_enable->integer != 0 && upscaler.model_loaded && upscaler.pipelines_ready;
}

// The source the spatial pass reads. On its own that is the tone-mapped TAA
// output; chained behind NSS it is NSS's reconstruction, which is already at
// twice the render extent.
static VkExtent2D spatial_source_extent(void)
{
	if (nss_is_active()) {
		VkExtent2D out = { nss.width * 2, nss.height * 2 };
		return out;
	}
	return qvk.extent_taa_output;
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

// Shared by both the spatial and NSS temporal paths: queries the shape,
// element type and byte size of one input/output tensor of `session`. Used
// with elem_type = UINT8 for the spatial models (whole tensor already
// quantized at the graph boundary) and FLOAT for NSS (whose QAT export bakes
// QuantizeLinear/DequantizeLinear inside the graph -- see
// baseq2/models/nss-temporal-high-int8.metadata.json).
static bool query_tensor_shape(OrtSession *session, bool is_input, size_t index, const char *expected_name,
	ONNXTensorElementDataType expected_elem_type, int64_t *dims_out, size_t num_dims_expected, size_t *byte_size_out)
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

		if (num_dims != num_dims_expected) {
			Com_EPrintf("upscaler: expected a %zu-D tensor, model %s %zu has %zu dims\n",
				num_dims_expected, is_input ? "input" : "output", index, num_dims);
			goto done;
		}

		if (!ort_ok(upscaler.api->GetDimensions(tensor_info, dims_out, num_dims), "GetDimensions"))
			goto done;

		ONNXTensorElementDataType elem_type;
		if (!ort_ok(upscaler.api->GetTensorElementType(tensor_info, &elem_type), "GetTensorElementType"))
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
// NSS's counterpart to the spatial NPU test below: same idea (flat synthetic
// input, repeated Run() for a fixed duration, report throughput), but against
// nss.session with its 1-input/2-output float32 contract instead of the
// spatial path's 1-input/1-output uint8 one.
static void nss_npu_test(int duration_ms)
{
	void *input_data = Z_Mallocz(nss.input_byte_size);
	void *kpn_data = Z_Mallocz(nss.kpn_byte_size);
	void *temporal_data = Z_Mallocz(nss.temporal_byte_size);

	OrtValue *input_value = NULL, *kpn_value = NULL, *temporal_value = NULL;
	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, input_data,
			nss.input_byte_size, nss.input_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_value), "CreateTensorWithDataAsOrtValue(nss input)"))
		goto done;

	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, kpn_data,
			nss.kpn_byte_size, nss.kpn_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &kpn_value), "CreateTensorWithDataAsOrtValue(nss kpn)"))
		goto done;

	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, temporal_data,
			nss.temporal_byte_size, nss.temporal_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &temporal_value), "CreateTensorWithDataAsOrtValue(nss temporal)"))
		goto done;

	{
		const char *input_names[]  = { NSS_INPUT_NAME };
		const char *output_names[] = { NSS_KPN_NAME, NSS_TEMPORAL_NAME };
		const OrtValue *inputs[]   = { input_value };
		OrtValue *outputs[]        = { kpn_value, temporal_value };

		Com_Printf("upscaler: running NSS NPU test inference for %d ms "
			"(watch Task Manager > Performance > NPU)\n", duration_ms);

		unsigned start = Sys_Milliseconds();
		unsigned elapsed = 0;
		int iterations = 0;
		while ((int)elapsed < duration_ms) {
			OrtStatus *status = upscaler.api->Run(nss.session, NULL,
				input_names, inputs, 1, output_names, 2, outputs);
			if (status) {
				Com_EPrintf("upscaler: NSS NPU test inference failed: %s\n", upscaler.api->GetErrorMessage(status));
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
	if (kpn_value)
		upscaler.api->ReleaseValue(kpn_value);
	if (temporal_value)
		upscaler.api->ReleaseValue(temporal_value);
	Z_Free(input_data);
	Z_Free(kpn_data);
	Z_Free(temporal_data);
}

static void Upscaler_NpuTest_f(void)
{
	if (!upscaler.model_loaded) {
		Com_Printf("upscaler: NPU upscaler model not loaded, nothing to test\n");
		return;
	}

	int duration_ms = (Cmd_Argc() > 1 ? atoi(Cmd_Argv(1)) : UPSCALER_DEFAULT_TEST_SECONDS) * 1000;
	if (duration_ms <= 0)
		duration_ms = UPSCALER_DEFAULT_TEST_SECONDS * 1000;

	// NSS's session/tensor contract is entirely different from the spatial
	// path below (1 input/2 float32 outputs vs. 1 input/1 uint8 output), so it
	// gets its own test. When both slots are loaded, test the spatial one and
	// leave NSS to a separate invocation -- timing them together would report a
	// number that is neither model's.
	if (nss.model_loaded && !upscaler.model_loaded) {
		nss_npu_test(duration_ms);
		return;
	}

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

// Writes one planar (NCHW, uint8, RGB) size x size tensor tile out as a PNG,
// for visually inspecting what actually goes into/comes out of the model.
// Triggered by the "upscaler_dump" console command via vkpt_upscaler_run_inference().
static void upscaler_dump_tensor(const char *stage, uint32_t tile, const uint8_t *planar, uint32_t size)
{
	uint8_t *interleaved = Z_Malloc((size_t)size * size * 3);

	size_t plane_stride = (size_t)size * size;
	for (uint32_t y = 0; y < size; y++) {
		for (uint32_t x = 0; x < size; x++) {
			size_t src = (size_t)y * size + x;
			size_t dst = ((size_t)y * size + x) * 3;
			interleaved[dst + 0] = planar[0 * plane_stride + src];
			interleaved[dst + 1] = planar[1 * plane_stride + src];
			interleaved[dst + 2] = planar[2 * plane_stride + src];
		}
	}

	char path[MAX_OSPATH];
	if (Q_snprintf(path, sizeof(path), "%s/screenshots/upscaler/upscaler_%s_%" PRIu64 "_tile%u.png",
			fs_gamedir, stage, qvk.frame_counter, tile) >= sizeof(path))
	{
		Com_EPrintf("upscaler: dump path too long\n");
		goto done;
	}

	if (FS_CreatePath(path) < 0) {
		Com_EPrintf("upscaler: failed to create directory for '%s'\n", path);
		goto done;
	}

	if (!stbi_write_png(path, size, size, 3, interleaved, size * 3))
		Com_EPrintf("upscaler: failed to write '%s'\n", path);
	else
		Com_Printf("upscaler: wrote %s\n", path);

done:
	Z_Free(interleaved);
}

// Writes one 3-channel group out of an NCHW float32 tensor as a PNG, for
// visually checking nss_pack.comp's input tensor. Values are assumed already
// in [0,1] display-referred range (true for the history/colour channels --
// see nss_pack.comp's header), so unlike upscaler_dump_tensor there is no
// uint8 quantization to undo, just a float->byte round. The KPN/temporal
// outputs are not dumped this way: their channels are per-tap filter weights
// and blend parameters, not an RGB image.
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
	upscaler.dump_requested = true;
	nss.dump_requested = true;
	Com_Printf("upscaler: will dump pre/post-upscale tensors for the next frame\n");
}

static const cmdreg_t upscaler_cmds[] = {
	{ "upscaler_npu_test", &Upscaler_NpuTest_f, NULL },
	{ "upscaler_dump", &Upscaler_Dump_f, NULL },
	{ NULL, NULL, NULL }
};

static void destroy_tensor_buffers(void); // defined with the render integration below

// The pack/unpack shaders index x and y with the same tile edge and pack 4
// pixels per dword, and the staging layout assumes 3 uint8 planes. Anything
// outside that is rejected rather than rendered as garbage.
static bool validate_tensor_geometry(const char *model_file)
{
	const int64_t *in = upscaler.input_dims, *out = upscaler.output_dims;

	if (in[0] != 1 || in[1] != 3 || out[0] != 1 || out[1] != 3) {
		Com_EPrintf("upscaler: %s is not a 1x3xNxN NCHW model "
			"(in [%lld,%lld,...], out [%lld,%lld,...]); not loading\n", model_file,
			(long long)in[0], (long long)in[1], (long long)out[0], (long long)out[1]);
		return false;
	}

	if (in[2] != in[3] || out[2] != out[3]) {
		Com_EPrintf("upscaler: %s tiles are not square (%lldx%lld -> %lldx%lld); not loading\n",
			model_file, (long long)in[2], (long long)in[3], (long long)out[2], (long long)out[3]);
		return false;
	}

	if (in[2] <= 0 || out[2] <= 0 || in[2] % 4 != 0 || out[2] % 4 != 0 || out[2] > UPSCALER_MAX_TILE) {
		Com_EPrintf("upscaler: %s tile size %lld -> %lld is unsupported "
			"(must be a positive multiple of 4, output at most %d); not loading\n",
			model_file, (long long)in[2], (long long)out[2], UPSCALER_MAX_TILE);
		return false;
	}

	// The scale factor drives the render extent (see get_render_extent), and the
	// unpack pass indexes the output grid as the render target scaled by it, so
	// it has to be a whole number.
	if (out[2] % in[2] != 0) {
		Com_EPrintf("upscaler: %s scale factor %lld -> %lld is not an integer; not loading\n",
			model_file, (long long)in[2], (long long)out[2]);
		return false;
	}

	upscaler.tile_in  = (uint32_t)in[2];
	upscaler.tile_out = (uint32_t)out[2];
	upscaler.scale    = upscaler.tile_out / upscaler.tile_in;
	return true;
}

// Shared by both the spatial and NSS temporal paths: session options + the
// QNN HTP execution provider + CreateSession, given a model file relative to
// the game dir. Both paths want the identical NPU deployment target -- see
// the plan's "target QNN HTP directly" decision -- so this is the one place
// that setup lives.
static bool create_qnn_session(const char *model_file, const char *display_name, OrtSession **out_session)
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

	char model_path[MAX_OSPATH];
	if (Q_concat(model_path, sizeof(model_path), fs_gamedir, PATH_SEP_STRING, model_file) >= sizeof(model_path)) {
		Com_EPrintf("upscaler: model path too long\n");
		upscaler.api->ReleaseSessionOptions(session_options);
		return false;
	}

	Com_Printf("upscaler: loading %s (%s), this takes a moment...\n", model_file, display_name);

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

// Releases the session for whichever model is loaded. The env and the CPU
// allocator outlive it, so switching models does not rebuild them.
static void unload_model(void)
{
	if (upscaler.session) {
		upscaler.api->ReleaseSession(upscaler.session);
		upscaler.session = NULL;
	}

	// The staging buffers are sized from the model's tile geometry, so they
	// cannot outlive it. The pack/unpack dispatches referencing them may still
	// be in flight, hence the wait.
	if (qvk.device)
		vkDeviceWaitIdle(qvk.device);
	destroy_tensor_buffers();

	upscaler.model_loaded = false;
	upscaler.loaded_model = 0;
	upscaler.tile_in = 0;
	upscaler.tile_out = 0;
	upscaler.scale = 0;
}

// Loads upscaler_models[index - 1]; index 0 just unloads. Creating the session
// is expensive -- the QNN EP finalizes the HTP graph, which takes seconds -- so
// this deliberately stalls rather than trying to hide the switch.
static void load_model(int index)
{
	if (!upscaler.initialized || index == upscaler.loaded_model)
		return;

	unload_model();

	if (index < 1 || index > (int)LENGTH(upscaler_models))
		return;

	const char *model_file = upscaler_models[index - 1].path;
	const char *display_name = upscaler_models[index - 1].name;

	// The temporal model has its own slot (nss_reload_model) and is never loaded
	// through here; the selector that drives this one already refuses it.
	if (upscaler_models[index - 1].kind == UPSCALER_KIND_TEMPORAL)
		return;

	if (!create_qnn_session(model_file, display_name, &upscaler.session))
		return;

	if (!query_tensor_shape(upscaler.session, true, 0, UPSCALER_INPUT_NAME, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
			upscaler.input_dims, UPSCALER_NUM_DIMS, &upscaler.input_byte_size) ||
		!query_tensor_shape(upscaler.session, false, 0, UPSCALER_OUTPUT_NAME, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
			upscaler.output_dims, UPSCALER_NUM_DIMS, &upscaler.output_byte_size) ||
		!validate_tensor_geometry(model_file))
	{
		unload_model();
		return;
	}

	upscaler.model_loaded = true;
	upscaler.loaded_model = index;

	Com_Printf("upscaler: %s loaded, %ux%u -> %ux%u per tile; "
		"run 'upscaler_npu_test' to verify NPU dispatch\n",
		display_name,
		upscaler.tile_in, upscaler.tile_in, upscaler.tile_out, upscaler.tile_out);
}

// Swaps the model and brings the pipelines back in line with it.
//
// The pipelines are normally built during VKPT_INIT_RELOAD_SHADER, which has
// already run by the time the user picks a model from the menu, and
// create_pipelines() no-ops when no model is loaded. So a model loaded this
// late has to rebuild them here; otherwise pipelines_ready stays false and
// vkpt_upscaler_is_enabled() -- which now also decides the render extent --
// would not become true until something else happened to reload shaders.
static void upscaler_load_model_and_pipelines(int index)
{
	int was_loaded = upscaler.loaded_model;

	load_model(index);

	// load_model() no-ops before ONNX Runtime is up and when the selection did
	// not change; either way the pipelines already match.
	if (upscaler.loaded_model == was_loaded || !qvk.device)
		return;

	vkpt_upscaler_destroy_pipelines();

	if (upscaler.model_loaded)
		vkpt_upscaler_create_pipelines();
}

// The temporal slot's counterpart to upscaler_load_model_and_pipelines(), with
// the same late-load rationale. Independent of the spatial slot: the two models
// occupy different stages of the frame and either, both, or neither may be up.
static void nss_load_model_and_pipelines(bool enable)
{
	int index = enable ? upscaler_temporal_model_index() : 0;

	if (!upscaler.initialized)
		return;
	if ((index != 0) == nss.model_loaded)
		return; // already in the requested state

	if (index == 0) {
		if (qvk.device)
			vkDeviceWaitIdle(qvk.device);
		nss_destroy_pipelines();
		nss_unload_model();
		return;
	}

	if (!nss_load_model(upscaler_models[index - 1].path, upscaler_models[index - 1].name))
		return;

	Com_Printf("upscaler: %s loaded, %ux%u fixed temporal render extent; "
		"run 'upscaler_npu_test' to verify NPU dispatch\n",
		upscaler_models[index - 1].name, nss.width, nss.height);

	if (qvk.device) {
		nss_destroy_pipelines();
		nss_create_pipelines();
	}
}

// Brings both slots in line with their selectors. Reached from the menu, via
// upscaling_mode_changed()/aa_mode_changed(), and from setting either backend
// cvar straight from the console.
static void upscaler_reload_model(void)
{
	upscaler_load_model_and_pipelines(cvar_flt_upscaler_enable->integer);
	nss_load_model_and_pipelines(cvar_flt_nss_enable->integer != 0);
}

static void upscaler_model_changed(cvar_t *self)
{
	upscaler_load_model_and_pipelines(self->integer);
}

static void nss_enable_changed(cvar_t *self)
{
	nss_load_model_and_pipelines(self->integer != 0);
}

VkResult vkpt_upscaler_initialize(void)
{
	memset(&upscaler, 0, sizeof(upscaler));
	memset(&nss, 0, sizeof(nss));

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

	if (!ort_ok(upscaler.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &upscaler.cpu_memory_info), "CreateCpuMemoryInfo")) {
		upscaler.api->ReleaseEnv(upscaler.env);
		upscaler.env = NULL;
		return VK_SUCCESS;
	}

	Cmd_Register(upscaler_cmds);

	upscaler.initialized = true;

	// Picking a different model from the menu or the console reloads the
	// session; init_cvars runs long before we exist, so the callbacks are only
	// hooked up here, and load_model() no-ops until initialized is set. Both
	// slots get one, so either can be driven straight from the console.
	cvar_flt_upscaler_enable->changed = upscaler_model_changed;
	cvar_flt_nss_enable->changed = nss_enable_changed;
	load_model(cvar_flt_upscaler_enable->integer);
	nss_load_model_and_pipelines(cvar_flt_nss_enable->integer != 0);

	return VK_SUCCESS;
}

VkResult vkpt_upscaler_destroy(void)
{
	unload_model();

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
// the descriptor set at them. The grid is derived from the render extent, so
// this also covers resolution changes, viewsize and dynamic render scaling.
static bool ensure_tensor_buffers(uint32_t tiles_x, uint32_t tiles_y)
{
	if (upscaler.tiles_x == tiles_x && upscaler.tiles_y == tiles_y && upscaler.input_mapped)
		return true;

	destroy_tensor_buffers();

	uint32_t num_tiles = tiles_x * tiles_y;
	VkDeviceSize input_size  = (VkDeviceSize)num_tiles * upscaler.input_byte_size;
	VkDeviceSize output_size = (VkDeviceSize)num_tiles * upscaler.output_byte_size;

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
	// Both slots build here when both are loaded; each no-ops when its model is
	// not up. destroy_pipelines() already tears down both unconditionally.
	if (nss.model_loaded)
		nss_create_pipelines();

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

// Tears down whichever pipelines are actually live, spatial and/or NSS,
// rather than branching on the currently *selected* kind: by the time this
// runs (from upscaler_load_model_and_pipelines(), after load_model() has
// already updated upscaler.loaded_model to the *new* selection), that would
// check the wrong kind and leak the outgoing model's pipelines. Every handle
// torn down below is null-checked, so calling both halves unconditionally is
// always safe -- whichever kind was not loaded already has null handles.
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

	nss_destroy_pipelines();

	return VK_SUCCESS;
}

bool vkpt_upscaler_is_enabled(void)
{
	return nss_is_active() || spatial_is_active();
}

// The model's fixed scale factor, or 0 when nothing is going to run. This is a
// property of the loaded model, not a resolution policy: the render extent comes
// from viewsize/DRS like every other path, and the factor only says how much
// bigger than that the model's output will be.
//
// For NSS this is nominal (2), used only by callers that just want an ">1 ==
// active" signal (e.g. main.c's upscaler_active_this_frame()); NSS's actual
// render extent is fixed, not derived from this value -- see
// vkpt_upscaler_get_temporal_extent().
uint32_t vkpt_upscaler_get_scale(void)
{
	if (nss_is_active())
		return 2;
	return spatial_is_active() ? upscaler.scale : 0;
}

bool vkpt_upscaler_get_temporal_extent(VkExtent2D *out)
{
	if (!nss_is_active())
		return false;
	out->width  = nss.width;
	out->height = nss.height;
	return true;
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

	VkExtent2D src = spatial_source_extent();

	upscaler_push_constants_t push = {
		upscaler.tiles_x, upscaler.tiles_y, upscaler.tile_in, upscaler.tile_out,
		src.width, src.height, nss_is_active() ? 1u : 0u
	};
	vkCmdPushConstants(cmd_buf, upscaler.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
		0, sizeof(push), &push);
}

// Records the spatial pack pass. Called from vkpt_upscaler_do() when the spatial
// model runs on its own, and from the chain command buffer in
// vkpt_upscaler_run_inference() when it runs behind NSS -- in that case the
// source is NSS's reconstruction, which does not exist yet at vkpt_upscaler_do()
// time because the NPU has not been asked for it.
static VkResult spatial_pack(VkCommandBuffer cmd_buf)
{
	upscaler.packed_this_frame = false;
	upscaler.tensor_valid = false;

	// The grid is sized from the source, not from the display: pack copies the
	// source into it 1:1, so it has to cover exactly what pack will read and one
	// input tile covers upscaler.tile_in pixels of it.
	//
	// The display does not participate at all. The model multiplies whatever it
	// is given by its fixed factor, so the grid covers source * scale, which is
	// >= the display whenever viewsize >= 100 / scale and the unpack pass
	// resolves the difference by downsampling. At viewsize 25 with a 128 -> 512
	// model this is 4x3 tiles at 1080p and the ratio is 1.0, i.e. the original
	// pinned-extent behaviour falls out as a special case.
	VkExtent2D src = spatial_source_extent();
	uint32_t tiles_x = (src.width  + upscaler.tile_in - 1) / upscaler.tile_in;
	uint32_t tiles_y = (src.height + upscaler.tile_in - 1) / upscaler.tile_in;

	if (tiles_x == 0 || tiles_y == 0)
		return VK_SUCCESS;

	// Refuse rather than try: the allocation below is host-visible and the
	// inference loop is serial, so an over-budget frame does not degrade, it
	// stalls for seconds or fails to map. Warn once per grid size so a config
	// that sits over the limit does not spam every frame.
	int max_tiles = cvar_flt_upscaler_max_tiles->integer;
	if (max_tiles > 0 && (int)(tiles_x * tiles_y) > max_tiles)
	{
		static uint32_t warned_for_tiles = 0;
		if (warned_for_tiles != tiles_x * tiles_y)
		{
			warned_for_tiles = tiles_x * tiles_y;
			Com_WPrintf("NPU upscaler: %ux%u = %u tiles exceeds flt_upscaler_max_tiles (%d); "
				"skipping. Lower viewsize or raise the limit.\n",
				tiles_x, tiles_y, tiles_x * tiles_y, max_tiles);
		}
		return VK_SUCCESS;
	}

	if (!ensure_tensor_buffers(tiles_x, tiles_y))
		return VK_SUCCESS;

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_PACK);

	bind_upscaler_pipeline(cmd_buf, upscaler.pipeline_pack);

	// The pack shader writes 4 horizontally adjacent pixels per invocation.
	uint32_t dispatch_x = tiles_x * (upscaler.tile_in / 4);
	uint32_t dispatch_y = tiles_y * upscaler.tile_in;
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

// The first pass of the frame's NPU work. NSS goes first when it is on, because
// the spatial model consumes its output; the spatial pack then cannot be
// recorded here and moves into the chain command buffer in
// vkpt_upscaler_run_inference().
VkResult vkpt_upscaler_do(VkCommandBuffer cmd_buf)
{
	if (nss_is_active())
		return nss_do(cmd_buf);

	if (spatial_is_active())
		return spatial_pack(cmd_buf);

	return VK_SUCCESS;
}

// Runs the NPU inference for the work vkpt_upscaler_do() packed. Must be called
// after the command buffer containing the pack dispatch has been submitted --
// see the call site in R_RenderFrame_RTX.
//
// This is a hard pipeline stall: the GPU has to finish the pack pass before the
// CPU can read the tensor, and the unpack pass can't be recorded until the NPU
// has produced the output. That round trip is inherent to driving the NPU from
// the CPU, and is the main reason per-frame NPU upscaling is only viable when
// the tile count stays low.
//
// Chaining NSS into the spatial model costs a second one: the spatial pack reads
// NSS's reconstruction, which does not exist until NSS's inference has returned
// and its reconstruct pass has run on the GPU. So that pair is recorded and
// submitted here, between the two inferences, rather than in main.c's frame
// graph -- which keeps the extra round trip contained to this file.
VkResult vkpt_upscaler_run_inference(void)
{
	if (nss_is_active())
	{
		VkResult res = nss_run_inference();

		if (!spatial_is_active())
			return res;

		// NSS produced nothing usable, so there is nothing to feed the spatial
		// stage. Clear its flags explicitly rather than just skipping it: they
		// still hold last frame's result, and the blit reads tensor_valid to
		// decide whether unpacking is safe.
		if (!nss.tensor_valid) {
			upscaler.packed_this_frame = false;
			upscaler.tensor_valid = false;
			return res;
		}

		VkCommandBuffer chain_cmd_buf = vkpt_begin_command_buffer(&qvk.cmd_buffers_graphics);

		nss_record_reconstruct(chain_cmd_buf);
		spatial_pack(chain_cmd_buf);

		vkpt_submit_command_buffer_simple(chain_cmd_buf, qvk.queue_graphics, true);

		// Falls through to the spatial inference below, which stalls on this
		// submission the same way the NSS one stalled on vkpt_upscaler_do()'s.
	}
	else if (!spatial_is_active())
	{
		return VK_SUCCESS;
	}

	if (!upscaler.packed_this_frame || !vkpt_upscaler_is_enabled())
		return VK_SUCCESS;

	upscaler.packed_this_frame = false;

	bool dump = upscaler.dump_requested;
	upscaler.dump_requested = false;

	vkQueueWaitIdle(qvk.queue_graphics);

	unsigned time_begin = Sys_Milliseconds();

	uint32_t num_tiles = upscaler.tiles_x * upscaler.tiles_y;
	const char *input_names[]  = { UPSCALER_INPUT_NAME };
	const char *output_names[] = { UPSCALER_OUTPUT_NAME };
	bool all_tiles_ok = true;

	for (uint32_t tile = 0; tile < num_tiles; tile++)
	{
		uint8_t *in  = (uint8_t *)upscaler.input_mapped  + (size_t)tile * upscaler.input_byte_size;
		uint8_t *out = (uint8_t *)upscaler.output_mapped + (size_t)tile * upscaler.output_byte_size;

		if (dump)
			upscaler_dump_tensor("pre", tile, in, upscaler.tile_in);

		OrtValue *input_value = NULL, *output_value = NULL;

		// Wrapping the mapped staging memory directly avoids an extra copy on
		// each side of the inference.
		if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, in,
				upscaler.input_byte_size, upscaler.input_dims, UPSCALER_NUM_DIMS,
				ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &input_value), "CreateTensorWithDataAsOrtValue(input)"))
			break;

		if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, out,
				upscaler.output_byte_size, upscaler.output_dims, UPSCALER_NUM_DIMS,
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
			all_tiles_ok = false;
			break;
		}

		if (dump)
			upscaler_dump_tensor("post", tile, out, upscaler.tile_out);
	}

	// Only a complete set of tiles is worth unpacking; a partial one would
	// present whatever the previous frame left in the staging buffer.
	upscaler.tensor_valid = all_tiles_ok;

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
	// The spatial model is always last when it runs, so it owns the blit. NSS
	// only reaches its own when it is the only one on -- chained, its
	// reconstruct pass has already run in the chain command buffer.
	if (!spatial_is_active())
		return nss_final_blit(cmd_buf, warp);

	// The pass can bail before producing anything -- a failed staging allocation
	// leaves the buffers destroyed, and a failed inference leaves the tensor
	// holding the previous frame. Unpacking either would present garbage or read
	// an already-freed buffer, so fall back to the best image that does exist:
	// NSS's reconstruction when it ran, otherwise the tone-mapped frame. Both are
	// below display resolution here, hence the filtered blit.
	if (!upscaler.tensor_valid) {
		if (nss_is_active() && nss.tensor_valid) {
			VkExtent2D nss_out = { nss.width * 2, nss.height * 2 };
			bool nss_needs_filter = nss_out.width  != qvk.extent_unscaled.width
			                     || nss_out.height != qvk.extent_unscaled.height;
			return vkpt_final_blit(cmd_buf, VKPT_IMG_NSS_OUTPUT, nss_out,
				nss_needs_filter, warp);
		}

		bool needs_filter = qvk.extent_taa_output.width  != qvk.extent_unscaled.width
		                 || qvk.extent_taa_output.height != qvk.extent_unscaled.height;
		return vkpt_final_blit(cmd_buf, VKPT_IMG_TAA_OUTPUT, qvk.extent_taa_output,
			needs_filter, warp);
	}

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);

	bind_upscaler_pipeline(cmd_buf, upscaler.pipeline_unpack);

	// Unpack writes only the display rect of IMG_UPSCALE_OUTPUT, matching what
	// the final blit below samples back out of it.
	VkExtent2D out = qvk.extent_unscaled;
	vkCmdDispatch(cmd_buf, (out.width + 7) / 8, (out.height + 7) / 8, 1);

	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_UPSCALE_OUTPUT]);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);

	return vkpt_final_blit(cmd_buf, VKPT_IMG_UPSCALE_OUTPUT, qvk.extent_unscaled, false, warp);
}

// ========================================================================== //
// NSS temporal path implementation. See the "NSS temporal path" comment above
// the nss struct definition for how this differs from the spatial path above,
// and nss_pack.comp/nss_reconstruct.comp for the actual algorithm.
// ========================================================================== //

static bool nss_load_model(const char *model_file, const char *display_name)
{
	if (!create_qnn_session(model_file, display_name, &nss.session))
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
			"dims a multiple of 8); not loading\n", model_file);
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
		upscaler.api->ReleaseSession(nss.session);
		nss.session = NULL;
	}

	if (qvk.device)
		vkDeviceWaitIdle(qvk.device);
	nss_destroy_tensor_buffers();

	nss.model_loaded = false;
	nss.width = 0;
	nss.height = 0;
}

// (Re)allocates the fixed-size tensor staging buffers and points the
// descriptor set at them. Unlike the spatial path's ensure_tensor_buffers(),
// there is no per-frame resize case: NSS's shape does not follow viewsize/DRS
// (see get_render_extent() in main.c), so this only ever needs to run once
// per model load, from nss_create_pipelines().
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
	nss.kpn_mapped       = buffer_map(&nss.buf_kpn);
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
	// extent straight off global_ubo, so unlike the spatial path this needs no
	// push constants.
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

// Single full-frame dispatch, unlike the spatial path's tile grid -- NSS's
// fixed shape already covers a normal render extent in one shot.
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

// Same hard-stall rationale as the spatial path's vkpt_upscaler_run_inference():
// must run after the command buffer holding nss_do()'s dispatch is submitted,
// and before nss_final_blit(). One inference for the whole frame rather than a
// serial per-tile loop.
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

	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, nss.input_mapped,
			nss.input_byte_size, nss.input_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_value), "CreateTensorWithDataAsOrtValue(nss input)"))
		goto done;

	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, nss.kpn_mapped,
			nss.kpn_byte_size, nss.kpn_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &kpn_value), "CreateTensorWithDataAsOrtValue(nss kpn)"))
		goto done;

	if (!ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info, nss.temporal_mapped,
			nss.temporal_byte_size, nss.temporal_dims, NSS_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &temporal_value), "CreateTensorWithDataAsOrtValue(nss temporal)"))
		goto done;

	{
		const char *input_names[]  = { NSS_INPUT_NAME };
		const char *output_names[] = { NSS_KPN_NAME, NSS_TEMPORAL_NAME };
		const OrtValue *inputs[]   = { input_value };
		OrtValue *outputs[]        = { kpn_value, temporal_value };

		OrtStatus *status = upscaler.api->Run(nss.session, NULL,
			input_names, inputs, 1, output_names, 2, outputs);

		if (status) {
			Com_EPrintf("upscaler: NSS inference failed: %s\n", upscaler.api->GetErrorMessage(status));
			upscaler.api->ReleaseStatus(status);
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
		upscaler.api->ReleaseValue(input_value);
	if (kpn_value)
		upscaler.api->ReleaseValue(kpn_value);
	if (temporal_value)
		upscaler.api->ReleaseValue(temporal_value);

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

// Records the reconstruct pass into VKPT_IMG_NSS_OUTPUT. Separate from
// nss_final_blit() because when the spatial model is chained behind NSS this has
// to run in the middle of the frame, in the chain command buffer, so its result
// is on the GPU in time for the spatial pack to read it.
//
// It writes a dedicated image rather than VKPT_IMG_UPSCALE_OUTPUT so the two
// stages do not alias: chained, the spatial unpack writes UPSCALE_OUTPUT while
// this is still being read as its source.
static void nss_record_reconstruct(VkCommandBuffer cmd_buf)
{
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);

	nss_bind_pipeline(cmd_buf, nss.pipeline_reconstruct);

	// Dispatches at the model's true output extent (render extent at an exact
	// 2x scale -- the only ratio the static KPN tap LUT is valid for, see
	// nss_reconstruct.comp's header) and writes that corner sub-rect in one pass
	// -- see nss_reconstruct.comp's header for why there is no separate
	// crop/downsample step the way the spatial path's unpack needs.
	VkExtent2D out = { nss.width * 2, nss.height * 2 };
	vkCmdDispatch(cmd_buf, (out.width + 7) / 8, (out.height + 7) / 8, 1);

	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_NSS_OUTPUT]);
	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_NSS_HISTORY_COLOR_A]);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_UNPACK);
}

static VkResult nss_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	// Same fallback rationale as the spatial path: a failed inference or a
	// pass that never ran this frame leaves nothing valid to reconstruct from.
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
