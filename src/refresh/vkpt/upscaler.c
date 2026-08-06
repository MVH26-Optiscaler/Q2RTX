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

	Q2RTX cvars
	-----------
	* flt_upscaler_enable - which model to run: 0 = disabled,
	  1 = QuickSRNetSmall, 2 = QuickSRNetLarge, 3 = QuickSRNetLarge fine-tuned on
	  Quake II RTX frames, 4 = the compact temporal model (see upscaler_models[]).
	  Changing it reloads the ONNX Runtime session, which takes a few seconds
	  because the QNN EP finalizes the HTP graph. Normally driven by the
	  flt_upscaling menu cvar.
	* flt_upscaler_max_tiles - refuse to run a frame needing more than this many
	  tiles, so an over-ambitious viewsize degrades to the non-upscaled blit
	  instead of stalling for seconds on a huge host-visible allocation.
	* flt_upscaler_max_bytes - the temporal path's equivalent budget, in MB.
	* flt_upscaler_reproj_threshold - tunes the temporal path's disocclusion test.
	* flt_upscaler_verbose - raise ONNX Runtime logging to verbose, which is
	  where per-node execution-provider assignment is reported.

	Temporal models
	---------------
	The fourth model is a different animal: three float32 inputs and two outputs,
	one inference for the whole frame, and a 16-channel latent state that it emits
	and the engine must reproject back to it every frame. It pins the render
	extent rather than scaling whatever viewsize picked, and it reads a pre-TAA
	tone-mapped frame (VKPT_IMG_UPSCALE_INPUT) rather than the TAA output, since
	it does that reconstruction itself.

	Its spatial dimensions are free in the ONNX graph, and get pinned to the
	display divided by the model's declared scale when the session is created --
	AddFreeDimensionOverrideByName, because the QNN EP cannot finalize a graph
	with free dimensions and would leave the work on the CPU. So the extent
	follows the display rather than the file, at the cost of rebuilding the
	session when the display changes; see vkpt_upscaler_check_display_extent().

	The state feedback falls out of the pipelining above for free. Frame N runs
	the inference for what frame N-1 packed, so the state comes back *before*
	frame N's pack is recorded -- which is exactly the one frame of history the
	model wants. See temporal_pack() and temporal_run_inference().

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
cvar_t *cvar_flt_upscaler_verbose = NULL;
cvar_t *cvar_flt_upscaler_max_tiles = NULL;
cvar_t *cvar_flt_upscaler_max_bytes = NULL;
cvar_t *cvar_flt_upscaler_reproj_threshold = NULL;

extern cvar_t *cvar_flt_fsr_enable; // owned by fsr.c, initialized just before us
extern cvar_t *scr_viewsize;        // owned by the client; clamped to 25..200

cvar_t *cvar_flt_upscaling = NULL;

// What kind of model an entry in upscaler_models[] is. The two kinds share the
// cvars, the menu selector and the pipelined round trip, and nothing else: the
// tensors, the shaders, the staging layout and the resolution policy all differ.
typedef enum {
	// Square fixed-shape 3-channel uint8 NCHW, one frame in, one frame out. The
	// frame is covered by a grid of tiles and each tile is a separate inference.
	// Scales whatever render extent viewsize/DRS picked.
	UPSCALER_KIND_SPATIAL,
	// Three float32 NCHW inputs and two outputs, covering the whole frame in one
	// inference. Reconstructs temporally from a latent state it emits itself and
	// the engine warps back to it, so it needs motion and the pre-TAA frame. Its
	// spatial dims are free in the graph and get pinned when the session is
	// created, from the display and viewsize -- so viewsize chooses the extent as
	// it does everywhere else, but changing it rebuilds the session rather than
	// taking effect on the next frame, and DRS cannot apply at all.
	UPSCALER_KIND_TEMPORAL,
} upscaler_kind_t;

// The NPU upscaler models, indexed by flt_upscaler_enable - 1. All ship in the
// repo under baseq2/models. The first two are stock w8a8 builds from Qualcomm AI
// Hub, the third a QuickSRNet Large fine-tuned on Quake II RTX frames and
// re-exported with its weights embedded, so it has no .data sidecar; the fourth
// is the temporal variant, exported with free spatial dimensions so it runs at
// whatever the display is.
//
// `scale` is meaningful for temporal models only, where it has to be declared
// rather than discovered: it caps viewsize at the model's 1:1 point, and that
// has to be known before the session exists to pin its free dimensions. It is
// not taken on trust -- validate_temporal_geometry() re-derives the scale from
// the shapes ORT reports back and rejects the model if the two disagree.
// Spatial models leave it 0 and keep discovering everything from their tile
// geometry.
//
// Order matters: it defines both the flt_upscaler_enable values and the
// flt_upscaling values below, so append rather than insert.
static const struct {
	const char     *name;
	const char     *path;
	upscaler_kind_t kind;
	uint32_t        scale;
} upscaler_models[] = {
	{ "QuickSRNet Small",                "models/quicksrnetsmall-w8a8.onnx",              UPSCALER_KIND_SPATIAL,  0 },
	{ "QuickSRNet Large",                "models/quicksrnetlarge-w8a8.onnx",              UPSCALER_KIND_SPATIAL,  0 },
	{ "QuickSRNet Large (Q2RTX-tuned)",  "models/quicksrnetlarge-q2rtx-w8a8.onnx",        UPSCALER_KIND_SPATIAL,  0 },
	{ "QuickSRNet Compact Temporal 2x",  "models/compact-temporal-2x-hardgate-w8a8.onnx", UPSCALER_KIND_TEMPORAL, 2 },
};

// Derived rather than hardcoded anywhere, so appending another temporal model
// needs no change outside the table above.
static bool upscaler_model_is_temporal(int index)
{
	return index >= 1 && index <= (int)LENGTH(upscaler_models)
		&& upscaler_models[index - 1].kind == UPSCALER_KIND_TEMPORAL;
}

// Menu-facing selector for the mutually exclusive upscalers. The per-backend
// cvars stay authoritative so existing configs, scripts and console use keep
// working; this just keeps them from being enabled at the same time.
enum {
	UPSCALING_MODE_NONE     = 0,
	UPSCALING_MODE_FSR      = 1,
	UPSCALING_MODE_AI_FIRST = 2, // 2 .. 2 + <number of models> - 1
};

// Loads whatever model flt_upscaler_enable now names. No-op until the ONNX
// Runtime side is up, and cheap when the selection did not actually change.
static void upscaler_reload_model(void);

static void upscaling_mode_changed(cvar_t *self)
{
	// flt_upscaler_enable is a 1-based model index rather than a boolean, so
	// everything that only asks "is the AI upscaler on?" still just tests it
	// against zero, and an archived flt_upscaler_enable 1 still means the
	// original model.
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

	// The temporal models' equivalent backstop. They run one inference over the
	// whole frame, so tiles say nothing about their cost; what does bite is the
	// staging, because a 16-channel float32 latent state is far larger than the
	// frame itself. It also scales with the display, since that is what decides
	// the model's extent: the two frames in flight want ~200 MB between them at
	// 1080p, ~350 MB at 1440p and ~800 MB at 2160p. So the default admits the
	// first two and refuses 4K, which is a real limit rather than a guard against
	// a mistake -- raise it if you have the memory. The allocation is reported at
	// load either way.
	cvar_flt_upscaler_max_bytes = Cvar_Get("flt_upscaler_max_bytes", "512", CVAR_ARCHIVE);

	// How far the reprojected view depth may disagree with what the motion vector
	// predicts, relative, before the pixel counts as disoccluded and its history
	// is dropped. Same order as the reprojection tests in the ASVGF passes.
	cvar_flt_upscaler_reproj_threshold = Cvar_Get("flt_upscaler_reproj_threshold", "0.05", CVAR_ARCHIVE);

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
#define UPSCALER_DEFAULT_TEST_SECONDS 3

// Sanity bound on the model's output tile, so a bogus model can't ask for an
// absurd staging allocation before anything else notices.
#define UPSCALER_MAX_TILE 4096

// The temporal models' tensors, in binding order. Unlike the spatial models,
// which are quantized at the graph boundary and so present uint8 there, these
// are int8-QDQ *inside* a float32 graph: the quantize/dequantize pairs sit after
// the inputs and before the outputs, so everything crossing the boundary is
// float. Querying them as uint8 is the mistake to avoid.
//
// "trusted" in the state's name describes where it enters, not what happens to
// it: the graph feeds it straight into a Conv without masking. But everything
// downstream of that Conv passes through a Mul by temporal_confidence -- the hard
// gate on the fusion output -- so the network does reject stale history on its
// own. upscaler_temporal_pack.comp still premasks; see the comment there.
#define TEMPORAL_NUM_INPUTS 3
#define TEMPORAL_NUM_OUTPUTS 2

enum {
	TEMPORAL_IN_COLOR,      // current_frame        [1,3,H,W]
	TEMPORAL_IN_STATE,      // trusted_warped_state [1,C,H,W]
	TEMPORAL_IN_CONFIDENCE, // temporal_confidence  [1,1,H,W]
};

enum {
	TEMPORAL_OUT_COLOR,    // upscaled_frame  [1,3,scale*H,scale*W]
	TEMPORAL_OUT_STATE,    // current_state   [1,C,H,W]
};

static const char *temporal_input_names[TEMPORAL_NUM_INPUTS] = {
	"current_frame", "trusted_warped_state", "temporal_confidence"
};

static const char *temporal_output_names[TEMPORAL_NUM_OUTPUTS] = {
	"upscaled_frame", "current_state"
};

// The dim params a temporal model declares for its free spatial dimensions. The
// QNN HTP execution provider cannot finalize a graph with free dimensions -- it
// would leave the work on the CPU -- so these are pinned before CreateSession.
// Overriding the two input dims is enough: ORT then infers the output dims,
// which carry their own dim params, concretely from them.
#define TEMPORAL_DIM_PARAM_WIDTH  "lr_width"
#define TEMPORAL_DIM_PARAM_HEIGHT "lr_height"

// Sanity bound on the low-res extent a temporal model may demand. It pins the
// render extent, so a bogus model would otherwise drag the whole frame graph
// with it.
#define TEMPORAL_MAX_EXTENT 8192
// And on the latent state, which dominates the staging allocation.
#define TEMPORAL_MAX_STATE_CHANNELS 64

typedef struct {
	uint32_t tiles_x;
	uint32_t tiles_y;
	uint32_t tile_in;  // model input  edge, pixels
	uint32_t tile_out; // model output edge, pixels
	// The rect of the source image the pack pass may read. Pushed rather than
	// taken from the UBO because the pack and the unpack that consumes it run a
	// frame apart, by which time the UBO has moved on -- see upscaler_slot_t.
	uint32_t src_width;
	uint32_t src_height;
} upscaler_push_constants_t;

// The temporal pack/unpack pair's geometry. Also captured per slot and replayed,
// for the same reason as above -- the model can be swapped out between the pack
// and the unpack that consumes it.
typedef struct {
	uint32_t lr_width;
	uint32_t lr_height;
	uint32_t out_width;
	uint32_t out_height;
	uint32_t state_channels;
	// Whether there is a previous state worth warping into this frame. Zero is
	// the network's own "no history" input, so clearing this is all a reset takes.
	uint32_t has_history;
	float    reproj_threshold;
} upscaler_temporal_push_constants_t;

// One frame's worth of the spatial GPU <-> NPU round trip. The round trip is
// pipelined rather than stalled on: frame N packs into slot N%2 and moves on,
// frame N+1 runs the inference for slot N%2 and unpacks it. So the displayed
// upscale is one frame old, and the CPU waits on pack_fence -- which the
// previous frame's submit already signalled -- instead of draining the queue.
//
// Two slots suffice:
//  - buf_input is GPU-written in frame N and CPU-read in frame N+1 under
//    pack_fence; the next GPU write to it is frame N+2's pack, recorded after
//    that read returned on the same thread.
//  - buf_output is CPU-written in frame N and read by the unpack submitted in
//    frame N; the next CPU write is frame N+2, and R_BeginFrame_RTX's wait on
//    fences_frame_sync[current_frame_index] has by then guaranteed all of
//    frame N completed. That is also why reallocating a slot's buffers on a
//    grid change is safe: the last GPU reference to them was frame N-2's.
typedef struct {
	BufferResource_t          buf_input;
	BufferResource_t          buf_output;
	void                     *input_mapped;
	void                     *output_mapped;
	VkDescriptorSet           desc_set;
	uint32_t                  tiles_x;
	uint32_t                  tiles_y;
	// Captured when the pack is recorded and replayed at unpack time: viewsize
	// and DRS can change the tile grid between the two frames, and the unpack
	// has to describe the grid it was packed with, not the current one.
	upscaler_push_constants_t push;
	// The temporal path's equivalent, and the extent its buffers were sized for.
	upscaler_temporal_push_constants_t temporal_push;
	uint32_t                  lr_width;
	uint32_t                  lr_height;
	// Signalled by whichever submit carries this slot's pack dispatch. The only
	// thing tracking "the pack finished" -- vkpt_submit_command_buffer_simple()
	// passes no fence.
	VkFence                   pack_fence;
	bool                      fence_pending; // handed to a submit, not waited on since
	bool                      packed;        // pack submitted, inference not run yet
	bool                      tensor_valid;  // inference produced a complete tile set
	bool                      dump;          // dump_requested, captured at pack time
} upscaler_slot_t;

struct
{
	const OrtApi  *api;
	OrtEnv        *env;
	OrtSession    *session;
	OrtMemoryInfo *cpu_memory_info;
	bool           initialized;  // env/allocator are up; safe to load models
	bool           model_loaded;
	int            loaded_model; // 1-based index into upscaler_models, 0 = none
	upscaler_kind_t kind;        // upscaler_models[loaded_model - 1].kind

	int64_t        input_dims[UPSCALER_NUM_DIMS];
	int64_t        output_dims[UPSCALER_NUM_DIMS];
	size_t         input_byte_size;  // uint8 tensor, 1 byte/element
	size_t         output_byte_size;
	uint32_t       tile_in;          // input_dims[2..3], validated square
	uint32_t       tile_out;         // output_dims[2..3], validated square
	uint32_t       scale;            // tile_out / tile_in, validated integer

	// Temporal models only. Every tensor is float32, so the byte sizes are
	// element counts times sizeof(float), and the two staging buffers hold the
	// four inputs and the two outputs back to back at fixed offsets -- which is
	// all it takes, since CreateTensorWithDataAsOrtValue will wrap any pointer.
	int64_t        temporal_input_dims[TEMPORAL_NUM_INPUTS][UPSCALER_NUM_DIMS];
	int64_t        temporal_output_dims[TEMPORAL_NUM_OUTPUTS][UPSCALER_NUM_DIMS];
	size_t         temporal_input_bytes[TEMPORAL_NUM_INPUTS];
	size_t         temporal_output_bytes[TEMPORAL_NUM_OUTPUTS];
	size_t         temporal_input_offsets[TEMPORAL_NUM_INPUTS];   // into buf_input
	size_t         temporal_output_offsets[TEMPORAL_NUM_OUTPUTS]; // into buf_output
	size_t         temporal_input_total;
	size_t         temporal_output_total;
	uint32_t       lr_width;         // the extent the model pins the render to
	uint32_t       lr_height;
	uint32_t       state_channels;
	// The extent vkpt_upscaler_check_render_extent() last saw wanted, and when it
	// first wanted it. Rebuilding the session costs seconds, and both inputs to
	// the extent -- the display and viewsize -- can move in a rapid burst (a
	// window drag, a slider drag), so a change has to hold still before it is
	// acted on. Zero width means nothing is pending.
	uint32_t       pending_lr_width;
	uint32_t       pending_lr_height;
	unsigned       pending_since_ms;

	// Render integration.
	bool                  pipelines_ready;
	VkPipeline            pipeline_pack;
	VkPipeline            pipeline_unpack;
	VkPipelineLayout      pipeline_layout;
	VkDescriptorSetLayout desc_set_layout;
	VkDescriptorPool      desc_pool;

	// Host-visible staging for the GPU <-> NPU round trip, one set per frame in
	// flight. Adreno is a unified-memory part, so the compute shaders
	// read/write these directly and the CPU maps them persistently -- no
	// separate device-local copy.
	upscaler_slot_t  slots[MAX_FRAMES_IN_FLIGHT];

	// One-shot flag set by the upscaler_dump console command, captured into the
	// slot by spatial_pack() so the pre/post pair covers the same frame.
	bool             dump_requested;

	// Rolling cost of the NPU round trip. Accumulated over many frames because
	// Sys_Milliseconds() is far too coarse to time a single frame's inference.
	// wait_ms_accum is the pack_fence wait alone -- what is left of the old
	// vkQueueWaitIdle stall now that the round trip is pipelined.
	unsigned         inference_ms_accum;
	unsigned         wait_ms_accum;
	unsigned         inference_frames;
} upscaler;

#define UPSCALER_TIMING_INTERVAL 100 // frames between timing reports

extern cvar_t *cvar_tm_enable; // UBO cvar owned by main.c

// Everything a model of either kind needs before it can run at all.
static bool upscaler_is_active(void)
{
	return cvar_flt_upscaler_enable->integer != 0 && upscaler.model_loaded && upscaler.pipelines_ready;
}

static bool temporal_is_active(void)
{
	// A temporal model consumes VKPT_IMG_UPSCALE_INPUT, which only exists
	// because vkpt_tone_mapping_record_cmd_buffer() writes it. With tone mapping
	// off nothing does, and there is no other display-referred [0,1] image in the
	// frame to fall back on -- TAA_OUTPUT would still be linear HDR. So the model
	// sits the frame out and the fallback blit takes over.
	return upscaler_is_active() && upscaler.kind == UPSCALER_KIND_TEMPORAL
		&& cvar_tm_enable->integer != 0;
}

static bool spatial_is_active(void)
{
	return upscaler_is_active() && upscaler.kind == UPSCALER_KIND_SPATIAL;
}

// The pipelined round trip indexes its staging by qvk.current_frame_index: the
// frame that packs owns `cur`, and the same frame runs the inference for -- and
// unpacks -- what the previous one left in `prev`.
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
// the frame that would have consumed a packed tensor does not run (menu mode,
// FSR taking over, accumulation rendering) or the buffers behind it are about
// to be rebuilt at a different geometry.
void vkpt_upscaler_discard(void)
{
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		upscaler.slots[i].packed = false;
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

// Queries the shape, element type and byte size of one input/output tensor of
// `session`. The models are quantized at the graph boundary, so elem_type is
// always UINT8.
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
// The temporal variant of the harness below: one inference over the whole frame
// with all five tensors, so the timing is directly comparable to what the render
// path will pay. Fed a flat mid-gray frame with no history, which is the same
// state the model starts every scene from.
static void temporal_npu_test(int duration_ms)
{
	void *input_data = Z_Mallocz(upscaler.temporal_input_total);
	void *output_data = Z_Mallocz(upscaler.temporal_output_total);

	// Mid-gray colour, everything else zero: no state and no confidence.
	float *color = (float *)input_data;
	size_t color_elements = upscaler.temporal_input_bytes[TEMPORAL_IN_COLOR] / sizeof(float);
	for (size_t i = 0; i < color_elements; i++)
		color[i] = 0.5f;

	OrtValue *inputs[TEMPORAL_NUM_INPUTS] = { NULL };
	OrtValue *outputs[TEMPORAL_NUM_OUTPUTS] = { NULL };
	bool ok = true;

	for (int i = 0; i < TEMPORAL_NUM_INPUTS && ok; i++) {
		ok = ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info,
			(uint8_t *)input_data + upscaler.temporal_input_offsets[i],
			upscaler.temporal_input_bytes[i], upscaler.temporal_input_dims[i], UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[i]), "CreateTensorWithDataAsOrtValue(input)");
	}

	for (int i = 0; i < TEMPORAL_NUM_OUTPUTS && ok; i++) {
		ok = ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info,
			(uint8_t *)output_data + upscaler.temporal_output_offsets[i],
			upscaler.temporal_output_bytes[i], upscaler.temporal_output_dims[i], UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputs[i]), "CreateTensorWithDataAsOrtValue(output)");
	}

	if (ok) {
		Com_Printf("upscaler: running temporal NPU test inference for %d ms "
			"(watch Task Manager > Performance > NPU)\n", duration_ms);

		unsigned start = Sys_Milliseconds();
		unsigned elapsed = 0;
		int iterations = 0;
		while ((int)elapsed < duration_ms) {
			OrtStatus *status = upscaler.api->Run(upscaler.session, NULL,
				temporal_input_names, (const OrtValue * const *)inputs, TEMPORAL_NUM_INPUTS,
				temporal_output_names, TEMPORAL_NUM_OUTPUTS, outputs);
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

	for (int i = 0; i < TEMPORAL_NUM_INPUTS; i++)
		if (inputs[i])
			upscaler.api->ReleaseValue(inputs[i]);
	for (int i = 0; i < TEMPORAL_NUM_OUTPUTS; i++)
		if (outputs[i])
			upscaler.api->ReleaseValue(outputs[i]);

	Z_Free(input_data);
	Z_Free(output_data);
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

	if (upscaler.kind == UPSCALER_KIND_TEMPORAL) {
		temporal_npu_test(duration_ms);
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

// Writes one planar (NCHW, uint8) size x size tensor tile out as a PNG, for
// visually inspecting what actually goes into/comes out of the model.
// Triggered by the "upscaler_dump" console command via spatial_run_inference().
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

// Opens a dump path under <gamedir>/screenshots/upscaler/ for the current frame.
// Returns false and complains if the path could not be produced.
static bool upscaler_dump_path(char *path, size_t size, const char *stage, const char *suffix)
{
	if (Q_snprintf(path, size, "%s/screenshots/upscaler/upscaler_%s_%" PRIu64 "%s.png",
			fs_gamedir, stage, qvk.frame_counter, suffix) >= size)
	{
		Com_EPrintf("upscaler: dump path too long\n");
		return false;
	}

	if (FS_CreatePath(path) < 0) {
		Com_EPrintf("upscaler: failed to create directory for '%s'\n", path);
		return false;
	}

	return true;
}

// The float32 equivalents of upscaler_dump_tensor(), for the temporal models.
// Values are assumed to be the [0,1] the model works in and are clamped rather
// than normalized, so a channel that has drifted out of range shows up as
// clipping instead of being quietly rescaled to look correct.
static void upscaler_dump_float_planes(const char *stage, const float *r, const float *g, const float *b,
	uint32_t width, uint32_t height)
{
	size_t count = (size_t)width * height;
	uint8_t *interleaved = Z_Malloc(count * 3);

	for (size_t i = 0; i < count; i++) {
		interleaved[i * 3 + 0] = (uint8_t)(min(max(r[i], 0.f), 1.f) * 255.f + 0.5f);
		interleaved[i * 3 + 1] = (uint8_t)(min(max(g[i], 0.f), 1.f) * 255.f + 0.5f);
		interleaved[i * 3 + 2] = (uint8_t)(min(max(b[i], 0.f), 1.f) * 255.f + 0.5f);
	}

	char path[MAX_OSPATH];
	if (upscaler_dump_path(path, sizeof(path), stage, "")) {
		if (!stbi_write_png(path, width, height, 3, interleaved, width * 3))
			Com_EPrintf("upscaler: failed to write '%s'\n", path);
		else
			Com_Printf("upscaler: wrote %s\n", path);
	}

	Z_Free(interleaved);
}

static void upscaler_dump_float_rgb(const char *stage, const float *r, const float *g, const float *b,
	uint32_t width, uint32_t height)
{
	upscaler_dump_float_planes(stage, r, g, b, width, height);
}

static void upscaler_dump_float_gray(const char *stage, const float *plane, uint32_t width, uint32_t height)
{
	upscaler_dump_float_planes(stage, plane, plane, plane, width, height);
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

// The temporal equivalent. Nothing here is square or tiled -- the model covers
// the frame in one inference -- but the three inputs have to agree on an extent,
// the state has to round-trip at the same channel count, and the whole thing has
// to be a whole-number upscale, because that extent becomes the render extent
// and the unpack indexes the output as a multiple of it.
static bool validate_temporal_geometry(const char *model_file, uint32_t want_lr_width, uint32_t want_lr_height,
	uint32_t want_scale)
{
	const int64_t *color    = upscaler.temporal_input_dims[TEMPORAL_IN_COLOR];
	const int64_t *state_in = upscaler.temporal_input_dims[TEMPORAL_IN_STATE];
	const int64_t *confidence = upscaler.temporal_input_dims[TEMPORAL_IN_CONFIDENCE];
	const int64_t *out      = upscaler.temporal_output_dims[TEMPORAL_OUT_COLOR];
	const int64_t *state_out= upscaler.temporal_output_dims[TEMPORAL_OUT_STATE];

	int64_t h = color[2], w = color[3];

	if (h <= 0 || w <= 0 || h > TEMPORAL_MAX_EXTENT || w > TEMPORAL_MAX_EXTENT) {
		Com_EPrintf("upscaler: %s low-res extent %lldx%lld is out of range (1..%d); not loading\n",
			model_file, (long long)w, (long long)h, TEMPORAL_MAX_EXTENT);
		return false;
	}

	// The render extent is rounded to an even width by get_render_extent(), and a
	// 2x pixel shuffle needs both axes even anyway.
	if ((w & 1) || (h & 1)) {
		Com_EPrintf("upscaler: %s low-res extent %lldx%lld is not even; not loading\n",
			model_file, (long long)w, (long long)h);
		return false;
	}

	if (color[0] != 1 || color[1] != 3 || out[0] != 1 || out[1] != 3) {
		Com_EPrintf("upscaler: %s colour tensors are not 1x3xHxW "
			"(in [%lld,%lld,...], out [%lld,%lld,...]); not loading\n", model_file,
			(long long)color[0], (long long)color[1], (long long)out[0], (long long)out[1]);
		return false;
	}

	if (confidence[1] != 1) {
		Com_EPrintf("upscaler: %s expects a 1-channel confidence mask, got %lld; not loading\n",
			model_file, (long long)confidence[1]);
		return false;
	}

	// Every input is sampled at the same pixel by the pack shader, so a
	// disagreement here would silently misindex rather than fail.
	for (int i = 0; i < TEMPORAL_NUM_INPUTS; i++) {
		const int64_t *d = upscaler.temporal_input_dims[i];
		if (d[0] != 1 || d[2] != h || d[3] != w) {
			Com_EPrintf("upscaler: %s input '%s' is [%lld,%lld,%lld,%lld], expected [1,*,%lld,%lld]; not loading\n",
				model_file, temporal_input_names[i],
				(long long)d[0], (long long)d[1], (long long)d[2], (long long)d[3],
				(long long)h, (long long)w);
			return false;
		}
	}

	// The state is a closed loop: what comes out is warped and fed straight back
	// in, so the two have to match exactly.
	if (state_out[0] != 1 || state_out[1] != state_in[1] || state_out[2] != h || state_out[3] != w) {
		Com_EPrintf("upscaler: %s state does not round-trip -- in [%lld,%lld,%lld,%lld], "
			"out [%lld,%lld,%lld,%lld]; not loading\n", model_file,
			(long long)state_in[0], (long long)state_in[1], (long long)state_in[2], (long long)state_in[3],
			(long long)state_out[0], (long long)state_out[1], (long long)state_out[2], (long long)state_out[3]);
		return false;
	}

	if (state_in[1] <= 0 || state_in[1] > TEMPORAL_MAX_STATE_CHANNELS) {
		Com_EPrintf("upscaler: %s wants %lld state channels, out of range (1..%d); not loading\n",
			model_file, (long long)state_in[1], TEMPORAL_MAX_STATE_CHANNELS);
		return false;
	}

	if (out[2] % h != 0 || out[3] % w != 0 || (out[2] / h) != (out[3] / w)) {
		Com_EPrintf("upscaler: %s scale factor %lldx%lld -> %lldx%lld is not a whole "
			"number in both axes; not loading\n", model_file,
			(long long)w, (long long)h, (long long)out[3], (long long)out[2]);
		return false;
	}

	// What the caller asked for when it pinned the free dimensions, read back
	// from the session. A mismatch means the model did not declare the dim
	// params we override, so its shapes are baked in and the extent we sized the
	// rest of the frame graph around is not the one it will actually run at.
	if ((uint32_t)w != want_lr_width || (uint32_t)h != want_lr_height) {
		Com_EPrintf("upscaler: %s runs at %lldx%lld, not the %ux%u it was asked for -- its spatial "
			"dimensions are not the free '%s'/'%s'; not loading\n", model_file,
			(long long)w, (long long)h, want_lr_width, want_lr_height,
			TEMPORAL_DIM_PARAM_WIDTH, TEMPORAL_DIM_PARAM_HEIGHT);
		return false;
	}

	// And the scale the model table declared, which is what turned the display
	// extent into the low-res one above. Discovering it disagrees now is too
	// late to fix, but not too late to refuse.
	if ((uint32_t)(out[2] / h) != want_scale) {
		Com_EPrintf("upscaler: %s upscales %ux, but the model table says %ux; not loading\n",
			model_file, (unsigned)(out[2] / h), want_scale);
		return false;
	}

	upscaler.lr_width       = (uint32_t)w;
	upscaler.lr_height      = (uint32_t)h;
	upscaler.state_channels = (uint32_t)state_in[1];
	upscaler.scale          = (uint32_t)(out[2] / h);

	// The staging layout: three inputs then two outputs, each tensor contiguous,
	// in the order the OrtValues are bound. Both shaders derive the same offsets
	// from the extents in their push constants, so this is the one place the
	// layout is decided.
	size_t offset = 0;
	for (int i = 0; i < TEMPORAL_NUM_INPUTS; i++) {
		upscaler.temporal_input_offsets[i] = offset;
		offset += upscaler.temporal_input_bytes[i];
	}
	upscaler.temporal_input_total = offset;

	offset = 0;
	for (int i = 0; i < TEMPORAL_NUM_OUTPUTS; i++) {
		upscaler.temporal_output_offsets[i] = offset;
		offset += upscaler.temporal_output_bytes[i];
	}
	upscaler.temporal_output_total = offset;

	return true;
}

// Session options + the QNN HTP execution provider + CreateSession, given a
// model file relative to the game dir.
//
// lr_width/lr_height pin the model's free spatial dimensions and are ignored for
// kinds that have none. They are what makes the render extent the caller's
// decision rather than the file's, so getting them wrong is not recoverable
// later: the QNN EP finalizes the graph at these dimensions during CreateSession.
static bool create_qnn_session(const char *model_file, const char *display_name, upscaler_kind_t kind,
	uint32_t lr_width, uint32_t lr_height, OrtSession **out_session)
{
	OrtSessionOptions *session_options = NULL;
	if (!ort_ok(upscaler.api->CreateSessionOptions(&session_options), "CreateSessionOptions"))
		return false;

	if (kind == UPSCALER_KIND_TEMPORAL)
	{
		// A model that turns out not to declare these is not an error here --
		// ORT ignores an override for a dim param it does not have, and a model
		// with its shapes already baked in simply reports them back unchanged,
		// which validate_temporal_geometry() then checks against the display the
		// same way. What is not tolerable is failing to set them on a model that
		// does declare them, hence the hard failure.
		if (!ort_ok(upscaler.api->AddFreeDimensionOverrideByName(session_options,
				TEMPORAL_DIM_PARAM_WIDTH, (int64_t)lr_width), "AddFreeDimensionOverrideByName(lr_width)") ||
			!ort_ok(upscaler.api->AddFreeDimensionOverrideByName(session_options,
				TEMPORAL_DIM_PARAM_HEIGHT, (int64_t)lr_height), "AddFreeDimensionOverrideByName(lr_height)"))
		{
			upscaler.api->ReleaseSessionOptions(session_options);
			return false;
		}
	}

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
	upscaler.kind = UPSCALER_KIND_SPATIAL;
	upscaler.tile_in = 0;
	upscaler.tile_out = 0;
	upscaler.scale = 0;
	upscaler.lr_width = 0;
	upscaler.lr_height = 0;
	upscaler.state_channels = 0;
	upscaler.pending_lr_width = 0;
}

// Rejects a model whose tensor count is not what its kind implies, so a stale or
// mismatched file produces a clean refusal here rather than an ORT error from
// indexing past the end of the session's tensor list.
static bool check_tensor_counts(const char *model_file, size_t want_inputs, size_t want_outputs)
{
	size_t num_inputs = 0, num_outputs = 0;

	if (!ort_ok(upscaler.api->SessionGetInputCount(upscaler.session, &num_inputs), "SessionGetInputCount") ||
		!ort_ok(upscaler.api->SessionGetOutputCount(upscaler.session, &num_outputs), "SessionGetOutputCount"))
		return false;

	if (num_inputs != want_inputs || num_outputs != want_outputs) {
		Com_EPrintf("upscaler: %s has %zu inputs and %zu outputs, expected %zu and %zu; not loading\n",
			model_file, num_inputs, num_outputs, want_inputs, want_outputs);
		return false;
	}

	return true;
}

// Queries and validates the five float32 tensors of a temporal model. The free
// dimensions were already pinned by create_qnn_session(), so ORT reports them
// concretely here -- including the outputs, which it infers from the inputs.
static bool load_temporal_tensors(const char *model_file, uint32_t want_lr_width, uint32_t want_lr_height,
	uint32_t want_scale)
{
	if (!check_tensor_counts(model_file, TEMPORAL_NUM_INPUTS, TEMPORAL_NUM_OUTPUTS))
		return false;

	for (int i = 0; i < TEMPORAL_NUM_INPUTS; i++) {
		if (!query_tensor_shape(upscaler.session, true, i, temporal_input_names[i],
				ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, upscaler.temporal_input_dims[i],
				UPSCALER_NUM_DIMS, &upscaler.temporal_input_bytes[i]))
			return false;
	}

	for (int i = 0; i < TEMPORAL_NUM_OUTPUTS; i++) {
		if (!query_tensor_shape(upscaler.session, false, i, temporal_output_names[i],
				ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, upscaler.temporal_output_dims[i],
				UPSCALER_NUM_DIMS, &upscaler.temporal_output_bytes[i]))
			return false;
	}

	return validate_temporal_geometry(model_file, want_lr_width, want_lr_height, want_scale);
}

// How long a wanted extent has to hold still before the session is rebuilt for
// it. Both inputs move in bursts -- dragging a window edge, dragging the
// viewsize slider -- and each rebuild costs seconds of QNN graph finalization,
// so acting on every intermediate value would be unusable.
#define TEMPORAL_EXTENT_SETTLE_MS 250

// The low-res extent a temporal model should run at: the display scaled by
// viewsize, rounded down to even, exactly as get_render_extent() does for every
// other path. False when there is nothing to compute it from yet.
//
// viewsize is capped at the model's 1:1 point -- 50 for a 2x model, where the
// output lands on the display exactly. Above that the model would render above
// the display resolution and the extra would be thrown away by the downsample,
// which for a whole-frame model means several times the staging for an image the
// display cannot show. The spatial models allow that because supersampling costs
// them only tiles; here it is gigabytes, and flt_upscaler_max_bytes would refuse
// it a moment later anyway. Capping keeps the stock viewsize 100 meaning "as
// good as this model gets" rather than "refused".
static bool temporal_desired_extent(uint32_t model_scale, uint32_t *out_width, uint32_t *out_height)
{
	if (qvk.extent_unscaled.width == 0 || qvk.extent_unscaled.height == 0 || model_scale < 1)
		return false;

	int scale = scr_viewsize ? scr_viewsize->integer : 100;
	int max_scale = (int)(100 / model_scale);

	if (scale > max_scale)
		scale = max_scale;
	if (scale < 1)
		return false;

	uint32_t w = (uint32_t)(qvk.extent_unscaled.width  * (float)scale / 100.f) & ~1u;
	uint32_t h = (uint32_t)(qvk.extent_unscaled.height * (float)scale / 100.f) & ~1u;

	if (w == 0 || h == 0)
		return false;

	*out_width  = w;
	*out_height = h;
	return true;
}

// The extent the loaded temporal model should be running at, if it is one.
static bool temporal_loaded_desired_extent(uint32_t *out_width, uint32_t *out_height)
{
	if (!upscaler.model_loaded || upscaler.kind != UPSCALER_KIND_TEMPORAL)
		return false;

	return temporal_desired_extent(upscaler_models[upscaler.loaded_model - 1].scale, out_width, out_height);
}

// True when the loaded temporal model is pinned to an extent that is no longer
// the one we want. Its free dimensions were fixed at session creation, so the
// only way to follow the display or viewsize is to build another session.
static bool temporal_session_extent_stale(void)
{
	uint32_t want_w, want_h;

	if (!temporal_loaded_desired_extent(&want_w, &want_h))
		return false;

	return want_w != upscaler.lr_width || want_h != upscaler.lr_height;
}

// Loads upscaler_models[index - 1]; index 0 just unloads. Creating the session
// is expensive -- the QNN EP finalizes the HTP graph, which takes seconds -- so
// this deliberately stalls rather than trying to hide the switch.
static void load_model(int index)
{
	if (!upscaler.initialized)
		return;

	// Reloading the same index is normally a no-op, but a temporal model built
	// for a display extent we no longer have is exactly that case and does need
	// the work -- see vkpt_upscaler_check_display_extent().
	if (index == upscaler.loaded_model && !temporal_session_extent_stale())
		return;

	unload_model();

	if (index < 1 || index > (int)LENGTH(upscaler_models))
		return;

	const char *model_file = upscaler_models[index - 1].path;
	const char *display_name = upscaler_models[index - 1].name;
	upscaler_kind_t kind = upscaler_models[index - 1].kind;
	uint32_t model_scale = upscaler_models[index - 1].scale;

	// A temporal model's low-res extent has to be settled before the session
	// exists, because pinning the free dimensions is part of creating one.
	uint32_t lr_width = 0, lr_height = 0;

	if (kind == UPSCALER_KIND_TEMPORAL && !temporal_desired_extent(model_scale, &lr_width, &lr_height)) {
		Com_EPrintf("upscaler: cannot size %s for a %ux%u display at viewsize %d; not loading\n",
			display_name, qvk.extent_unscaled.width, qvk.extent_unscaled.height,
			scr_viewsize ? scr_viewsize->integer : 0);
		return;
	}

	if (!create_qnn_session(model_file, display_name, kind, lr_width, lr_height, &upscaler.session))
		return;

	upscaler.kind = kind;

	if (kind == UPSCALER_KIND_TEMPORAL)
	{
		if (!load_temporal_tensors(model_file, lr_width, lr_height, model_scale)) {
			unload_model();
			return;
		}
	}
	else
	{
		if (!check_tensor_counts(model_file, 1, 1) ||
			!query_tensor_shape(upscaler.session, true, 0, UPSCALER_INPUT_NAME, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
				upscaler.input_dims, UPSCALER_NUM_DIMS, &upscaler.input_byte_size) ||
			!query_tensor_shape(upscaler.session, false, 0, UPSCALER_OUTPUT_NAME, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
				upscaler.output_dims, UPSCALER_NUM_DIMS, &upscaler.output_byte_size) ||
			!validate_tensor_geometry(model_file))
		{
			unload_model();
			return;
		}
	}

	upscaler.model_loaded = true;
	upscaler.loaded_model = index;

	if (kind == UPSCALER_KIND_TEMPORAL)
	{
		Com_Printf("upscaler: %s loaded, %ux%u -> %ux%u whole-frame, %u state channels, "
			"for a %ux%u display; the extent is pinned into the session, so changing viewsize "
			"or the resolution reloads it and DRS does not apply. "
			"Run 'upscaler_npu_test' to verify NPU dispatch\n",
			display_name,
			upscaler.lr_width, upscaler.lr_height,
			upscaler.lr_width * upscaler.scale, upscaler.lr_height * upscaler.scale,
			upscaler.state_channels,
			qvk.extent_unscaled.width, qvk.extent_unscaled.height);
	}
	else
	{
		Com_Printf("upscaler: %s loaded, %ux%u -> %ux%u per tile; "
			"run 'upscaler_npu_test' to verify NPU dispatch\n",
			display_name,
			upscaler.tile_in, upscaler.tile_in, upscaler.tile_out, upscaler.tile_out);
	}
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

// Brings the model in line with its selector. Reached from the menu, via
// upscaling_mode_changed()/aa_mode_changed(), and from setting either backend
// cvar straight from the console.
static void upscaler_reload_model(void)
{
	upscaler_load_model_and_pipelines(cvar_flt_upscaler_enable->integer);
}

static void upscaler_model_changed(cvar_t *self)
{
	upscaler_load_model_and_pipelines(self->integer);
}

// Keeps a temporal model's pinned extent in step with the display and viewsize.
//
// The extent is frozen into the session when the QNN EP finalizes the graph, so
// following either input means building another session, which stalls for
// seconds. Two consequences shape this.
//
// It compares the wanted extent rather than reacting to recreate_swapchain():
// alt-tab, minimize/restore, HDR and vsync changes all recreate the swapchain
// without moving the extent, and none of them should pay for a reload.
//
// And it waits for the extent to hold still. viewsize moves in 5% steps under a
// dragged slider and the display extent moves continuously under a dragged
// window edge; rebuilding on each intermediate value would stall for the whole
// drag. Nothing is displayed through the model in the meantime -- the previous
// session keeps running at its own extent until the new one is ready.
//
// Called from R_BeginFrame_RTX once the swapchain has settled for the frame, so
// get_render_extent() sees the new low-res extent immediately and the screen
// images are rebuilt by the check already there.
void vkpt_upscaler_check_render_extent(void)
{
	uint32_t want_w, want_h;

	if (!temporal_loaded_desired_extent(&want_w, &want_h))
		return;

	if (want_w == upscaler.lr_width && want_h == upscaler.lr_height) {
		upscaler.pending_lr_width = 0;
		return;
	}

	unsigned now = Sys_Milliseconds();

	// A different target than last frame restarts the clock, so a drag only
	// settles once the user stops moving it.
	if (upscaler.pending_lr_width != want_w || upscaler.pending_lr_height != want_h) {
		upscaler.pending_lr_width  = want_w;
		upscaler.pending_lr_height = want_h;
		upscaler.pending_since_ms  = now;
		return;
	}

	if (now - upscaler.pending_since_ms < TEMPORAL_EXTENT_SETTLE_MS)
		return;

	upscaler.pending_lr_width = 0;

	Com_Printf("upscaler: render extent is now %ux%u (%ux%u display at viewsize %d), "
		"rebuilding the model for it (this takes a moment)\n",
		want_w, want_h, qvk.extent_unscaled.width, qvk.extent_unscaled.height,
		scr_viewsize ? scr_viewsize->integer : 0);

	// Same index; load_model() consults temporal_session_extent_stale() rather
	// than no-opping on the match.
	upscaler_load_model_and_pipelines(cvar_flt_upscaler_enable->integer);
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

	if (!ort_ok(upscaler.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &upscaler.cpu_memory_info), "CreateCpuMemoryInfo")) {
		upscaler.api->ReleaseEnv(upscaler.env);
		upscaler.env = NULL;
		return VK_SUCCESS;
	}

	Cmd_Register(upscaler_cmds);

	upscaler.initialized = true;

	// Picking a different model from the menu or the console reloads the
	// session; init_cvars runs long before we exist, so the callback is only
	// hooked up here, and load_model() no-ops until initialized is set.
	cvar_flt_upscaler_enable->changed = upscaler_model_changed;
	load_model(cvar_flt_upscaler_enable->integer);

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

// Releases one slot's staging buffers. Safe to call when they were never
// allocated.
static void destroy_slot_buffers(upscaler_slot_t *slot)
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

	slot->tiles_x = 0;
	slot->tiles_y = 0;
	slot->lr_width = 0;
	slot->lr_height = 0;
	slot->packed = false;
	slot->tensor_valid = false;
}

// Releases every slot's staging buffers.
static void destroy_tensor_buffers(void)
{
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		destroy_slot_buffers(&upscaler.slots[i]);
}

// (Re)allocates one slot's tensor staging buffers for the current tile grid and
// points that slot's descriptor set at them. The grid is derived from the render
// extent, so this also covers resolution changes, viewsize and dynamic render
// scaling.
//
// Reallocating mid-frame is safe because only the packing frame's own slot is
// touched, and the last GPU work to reference it was frame N-2's unpack, which
// R_BeginFrame_RTX's fence wait has already accounted for.
static bool ensure_tensor_buffers(upscaler_slot_t *slot, uint32_t tiles_x, uint32_t tiles_y)
{
	if (slot->tiles_x == tiles_x && slot->tiles_y == tiles_y && slot->input_mapped)
		return true;

	destroy_slot_buffers(slot);

	uint32_t num_tiles = tiles_x * tiles_y;
	VkDeviceSize input_size  = (VkDeviceSize)num_tiles * upscaler.input_byte_size;
	VkDeviceSize output_size = (VkDeviceSize)num_tiles * upscaler.output_byte_size;

	const VkMemoryPropertyFlags host_props =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	if (buffer_create(&slot->buf_input, input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS ||
		buffer_create(&slot->buf_output, output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS)
	{
		Com_EPrintf("upscaler: failed to allocate %ux%u tile tensor buffers\n", tiles_x, tiles_y);
		destroy_slot_buffers(slot);
		return false;
	}

	buffer_attach_name(&slot->buf_input, "upscaler input tensor");
	buffer_attach_name(&slot->buf_output, "upscaler output tensor");

	slot->input_mapped  = buffer_map(&slot->buf_input);
	slot->output_mapped = buffer_map(&slot->buf_output);

	if (!slot->input_mapped || !slot->output_mapped) {
		Com_EPrintf("upscaler: failed to map tensor buffers\n");
		destroy_slot_buffers(slot);
		return false;
	}

	VkDescriptorBufferInfo buffer_info[] = {
		{ .buffer = slot->buf_input.buffer,  .offset = 0, .range = input_size  },
		{ .buffer = slot->buf_output.buffer, .offset = 0, .range = output_size },
	};

	VkWriteDescriptorSet writes[] = {
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = slot->desc_set,
			.dstBinding      = 0,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo     = &buffer_info[0],
		},
		{
			.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet          = slot->desc_set,
			.dstBinding      = 1,
			.descriptorCount = 1,
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo     = &buffer_info[1],
		},
	};
	vkUpdateDescriptorSets(qvk.device, LENGTH(writes), writes, 0, NULL);

	slot->tiles_x = tiles_x;
	slot->tiles_y = tiles_y;

	Com_Printf("upscaler: %ux%u tile grid (%u inferences/frame, %.1f MB staging per frame in flight)\n",
		tiles_x, tiles_y, num_tiles, (double)(input_size + output_size) / (1024.0 * 1024.0));

	return true;
}

// The slot whose output holds the state the given slot's pack must warp: the one
// packed a frame earlier. With MAX_FRAMES_IN_FLIGHT slots this is a fixed
// pairing, which is why it can be baked into the descriptor sets rather than
// rebound every frame.
static int temporal_prev_slot_index(int index)
{
	return (index + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
}

// (Re)allocates the temporal staging for *every* slot and wires up the
// descriptor sets, including the cross-slot binding each pack reads the previous
// state through. All slots at once rather than lazily per slot, because those
// cross references mean a half-allocated set is not a usable state, and because
// the extent is pinned when the session is created there is nothing to allocate
// lazily for -- it changes only when the model is (re)loaded, which is also what
// a display resolution change goes through.
//
// The descriptor sets themselves are allocated once with the pipelines and only
// rewritten here, so going round this repeatedly does not drain the pool.
static bool ensure_temporal_buffers(void)
{
	uint32_t lr_w = upscaler.lr_width, lr_h = upscaler.lr_height;

	if (upscaler.slots[0].lr_width == lr_w && upscaler.slots[0].lr_height == lr_h && upscaler.slots[0].input_mapped)
		return true;

	destroy_tensor_buffers();

	VkDeviceSize input_size  = (VkDeviceSize)upscaler.temporal_input_total;
	VkDeviceSize output_size = (VkDeviceSize)upscaler.temporal_output_total;

	double total_mb = (double)(input_size + output_size) * MAX_FRAMES_IN_FLIGHT / (1024.0 * 1024.0);
	int max_mb = cvar_flt_upscaler_max_bytes->integer;

	if (max_mb > 0 && total_mb > (double)max_mb) {
		static uint32_t warned_for_extent = 0;
		if (warned_for_extent != (lr_w << 16 | lr_h)) {
			warned_for_extent = lr_w << 16 | lr_h;
			Com_WPrintf("upscaler: %ux%u with %u state channels needs %.0f MB of staging, over "
				"flt_upscaler_max_bytes (%d MB); skipping. The extent follows the display, so "
				"raise the limit or run at a lower resolution.\n",
				lr_w, lr_h, upscaler.state_channels, total_mb, max_mb);
		}
		return false;
	}

	const VkMemoryPropertyFlags host_props =
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		upscaler_slot_t *slot = &upscaler.slots[i];

		if (buffer_create(&slot->buf_input, input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS ||
			buffer_create(&slot->buf_output, output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_props) != VK_SUCCESS)
		{
			Com_EPrintf("upscaler: failed to allocate %.0f MB of temporal tensor buffers\n", total_mb);
			destroy_tensor_buffers();
			return false;
		}

		buffer_attach_name(&slot->buf_input, "upscaler temporal input tensor");
		buffer_attach_name(&slot->buf_output, "upscaler temporal output tensor");

		slot->input_mapped  = buffer_map(&slot->buf_input);
		slot->output_mapped = buffer_map(&slot->buf_output);

		if (!slot->input_mapped || !slot->output_mapped) {
			Com_EPrintf("upscaler: failed to map temporal tensor buffers\n");
			destroy_tensor_buffers();
			return false;
		}

		// Freshly allocated device memory is not guaranteed to be anything in
		// particular, and the first pack reads a slot's output as state before
		// any inference has written it. Zero is the network's own "no history"
		// value, so this is the correct initial content rather than merely a
		// safe one -- and it keeps a NaN out of the warp.
		memset(slot->output_mapped, 0, upscaler.temporal_output_total);

		slot->lr_width  = lr_w;
		slot->lr_height = lr_h;
	}

	// Second pass: every buffer exists now, so the cross-slot references resolve.
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		upscaler_slot_t *slot = &upscaler.slots[i];
		upscaler_slot_t *prev = &upscaler.slots[temporal_prev_slot_index(i)];

		VkDescriptorBufferInfo buffer_info[] = {
			{ .buffer = slot->buf_input.buffer,  .offset = 0, .range = input_size  },
			{ .buffer = slot->buf_output.buffer, .offset = 0, .range = output_size },
			{ .buffer = prev->buf_output.buffer, .offset = 0, .range = output_size },
		};

		VkWriteDescriptorSet writes[LENGTH(buffer_info)];
		for (uint32_t b = 0; b < LENGTH(buffer_info); b++) {
			writes[b] = (VkWriteDescriptorSet){
				.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet          = slot->desc_set,
				.dstBinding      = b,
				.descriptorCount = 1,
				.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo     = &buffer_info[b],
			};
		}
		vkUpdateDescriptorSets(qvk.device, LENGTH(writes), writes, 0, NULL);
	}

	Com_Printf("upscaler: %ux%u temporal staging, %.1f MB per frame in flight (%.0f MB total)\n",
		lr_w, lr_h, (double)(input_size + output_size) / (1024.0 * 1024.0), total_mb);

	return true;
}

VkResult vkpt_upscaler_create_pipelines(void)
{
	if (!upscaler.model_loaded)
		return VK_SUCCESS;

	bool temporal = upscaler.kind == UPSCALER_KIND_TEMPORAL;

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
		{
			.binding         = 2, // previous slot's output, read by the temporal
			.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // pack for the state
			.descriptorCount = 1,
			.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
		},
	};

	// The third binding only exists for the temporal path, and only its pack
	// shader declares it.
	uint32_t num_bindings = temporal ? LENGTH(bindings) : 2;

	VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = num_bindings,
		.pBindings    = bindings,
	};
	_VK(vkCreateDescriptorSetLayout(qvk.device, &layout_info, NULL, &upscaler.desc_set_layout));
	ATTACH_LABEL_VARIABLE(upscaler.desc_set_layout, DESCRIPTOR_SET_LAYOUT);

	// One set per frame in flight: the frame that packs and the frame that
	// unpacks bind different staging buffers, so they cannot share one.
	VkDescriptorPoolSize pool_size = {
		.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = num_bindings * MAX_FRAMES_IN_FLIGHT,
	};
	VkDescriptorPoolCreateInfo pool_info = {
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets       = MAX_FRAMES_IN_FLIGHT,
		.poolSizeCount = 1,
		.pPoolSizes    = &pool_size,
	};
	_VK(vkCreateDescriptorPool(qvk.device, &pool_info, NULL, &upscaler.desc_pool));

	VkDescriptorSetLayout set_layouts[MAX_FRAMES_IN_FLIGHT];
	VkDescriptorSet       desc_sets[MAX_FRAMES_IN_FLIGHT];
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		set_layouts[i] = upscaler.desc_set_layout;

	VkDescriptorSetAllocateInfo alloc_info = {
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool     = upscaler.desc_pool,
		.descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
		.pSetLayouts        = set_layouts,
	};
	_VK(vkAllocateDescriptorSets(qvk.device, &alloc_info, desc_sets));

	// Created unsignalled: every wait is gated on the slot's `packed` flag, and
	// nothing sets that until a submit has been given the fence to signal.
	VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		upscaler.slots[i].desc_set = desc_sets[i];
		if (!upscaler.slots[i].pack_fence)
			_VK(vkCreateFence(qvk.device, &fence_info, NULL, &upscaler.slots[i].pack_fence));
	}
	vkpt_upscaler_discard();

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
		.size       = temporal ? sizeof(upscaler_temporal_push_constants_t)
		                       : sizeof(upscaler_push_constants_t),
	};

	CREATE_PIPELINE_LAYOUT(qvk.device, &upscaler.pipeline_layout,
		.setLayoutCount         = LENGTH(desc_set_layouts),
		.pSetLayouts            = desc_set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &push_constant_range,
	);
	ATTACH_LABEL_VARIABLE(upscaler.pipeline_layout, PIPELINE_LAYOUT);

	// SHADER_STAGE expands to a braced initializer rather than an expression, so
	// the kind is selected here, on the module, rather than on the stage.
	enum QVK_SHADER_MODULES mod_pack = temporal
		? QVK_MOD_UPSCALER_TEMPORAL_PACK_COMP : QVK_MOD_UPSCALER_PACK_COMP;
	enum QVK_SHADER_MODULES mod_unpack = temporal
		? QVK_MOD_UPSCALER_TEMPORAL_UNPACK_COMP : QVK_MOD_UPSCALER_UNPACK_COMP;

	VkComputePipelineCreateInfo pipeline_info[] = {
		{
			.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage  = SHADER_STAGE(mod_pack, VK_SHADER_STAGE_COMPUTE_BIT),
			.layout = upscaler.pipeline_layout,
		},
		{
			.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage  = SHADER_STAGE(mod_unpack, VK_SHADER_STAGE_COMPUTE_BIT),
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

// Tears down whatever is actually live rather than branching on the currently
// selected model: by the time this runs (from
// upscaler_load_model_and_pipelines(), after load_model() has already updated
// upscaler.loaded_model to the *new* selection) that would leak the outgoing
// model's pipelines. Every handle torn down below is null-checked.
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
	}
	// Every caller has already brought the device idle (see the comment above),
	// so no fence destroyed here can still be pending.
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		upscaler.slots[i].desc_set = VK_NULL_HANDLE;
		upscaler.slots[i].fence_pending = false;
		if (upscaler.slots[i].pack_fence) {
			vkDestroyFence(qvk.device, upscaler.slots[i].pack_fence, NULL);
			upscaler.slots[i].pack_fence = VK_NULL_HANDLE;
		}
	}
	if (upscaler.desc_set_layout) {
		vkDestroyDescriptorSetLayout(qvk.device, upscaler.desc_set_layout, NULL);
		upscaler.desc_set_layout = VK_NULL_HANDLE;
	}

	return VK_SUCCESS;
}

bool vkpt_upscaler_is_enabled(void)
{
	return upscaler_is_active();
}

bool vkpt_upscaler_wants_input_tap(void)
{
	// Deliberately not temporal_is_active(): that one is gated on tone mapping
	// being on, and this is what tone mapping asks to decide whether to write the
	// tap in the first place. Asking the stricter question here would be circular.
	return upscaler_is_active() && upscaler.kind == UPSCALER_KIND_TEMPORAL;
}

bool vkpt_upscaler_get_temporal_extent(VkExtent2D *extent)
{
	if (!temporal_is_active())
		return false;

	extent->width  = upscaler.lr_width;
	extent->height = upscaler.lr_height;
	return true;
}

// The model's fixed scale factor, or 0 when nothing is going to run.
//
// For a spatial model this is a property of the model, not a resolution policy:
// the render extent comes from viewsize/DRS like every other path, and the
// factor only says how much bigger than that the model's output will be. For a
// temporal model the causality is reversed -- its shape is baked into the graph,
// so it dictates the render extent and the factor merely describes the result.
uint32_t vkpt_upscaler_get_scale(void)
{
	return upscaler_is_active() && (upscaler.kind != UPSCALER_KIND_TEMPORAL || temporal_is_active())
		? upscaler.scale : 0;
}

// Binds one of the two pipelines against `slot`'s staging buffers, and
// pushes the tile geometry recorded in that slot. The geometry is replayed from
// the slot rather than recomputed because the unpack runs a frame after the pack
// that established it, by which time viewsize or DRS may have moved the grid on.
static void bind_upscaler_pipeline(VkCommandBuffer cmd_buf, VkPipeline pipeline, const upscaler_slot_t *slot)
{
	VkDescriptorSet desc_sets[] = {
		qvk.desc_set_ubo,
		qvk_get_current_desc_set_textures(),
		slot->desc_set,
	};

	vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
		upscaler.pipeline_layout, 0, LENGTH(desc_sets), desc_sets, 0, NULL);

	// The two kinds have different push constants, and the pipeline layout was
	// built for whichever kind is loaded.
	if (upscaler.kind == UPSCALER_KIND_TEMPORAL) {
		vkCmdPushConstants(cmd_buf, upscaler.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
			0, sizeof(slot->temporal_push), &slot->temporal_push);
	} else {
		vkCmdPushConstants(cmd_buf, upscaler.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
			0, sizeof(slot->push), &slot->push);
	}
}

// Records the pack pass into `slot`, from vkpt_upscaler_do().
static VkResult spatial_pack(VkCommandBuffer cmd_buf, upscaler_slot_t *slot)
{
	slot->packed = false;

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
	VkExtent2D src = qvk.extent_taa_output;
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

	if (!ensure_tensor_buffers(slot, tiles_x, tiles_y))
		return VK_SUCCESS;

	slot->push = (upscaler_push_constants_t){
		tiles_x, tiles_y, upscaler.tile_in, upscaler.tile_out,
		src.width, src.height
	};

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_PACK);

	bind_upscaler_pipeline(cmd_buf, upscaler.pipeline_pack, slot);

	// The pack shader writes 4 horizontally adjacent pixels per invocation.
	uint32_t dispatch_x = tiles_x * (upscaler.tile_in / 4);
	uint32_t dispatch_y = tiles_y * upscaler.tile_in;
	vkCmdDispatch(cmd_buf, (dispatch_x + 7) / 8, (dispatch_y + 7) / 8, 1);

	// Make the shader writes visible to the host read that next frame's
	// spatial_run_inference() will do once pack_fence reports this dispatch done.
	BUFFER_BARRIER(cmd_buf,
		.buffer        = slot->buf_input.buffer,
		.offset        = 0,
		.size          = VK_WHOLE_SIZE,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_PACK);
	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);

	// The fence is signalled by the frame's post submit, which is the command
	// buffer this pass records into -- see vkpt_upscaler_pack_fence().
	//
	// It can still be pending here if the frame that would have waited on it
	// never ran (vkpt_upscaler_discard()), and resetting a pending fence is
	// invalid. Draining it first costs nothing: that submit is at least two
	// frames old, so R_BeginFrame_RTX's own fence wait has already covered it.
	if (slot->fence_pending)
		_VK(vkWaitForFences(qvk.device, 1, &slot->pack_fence, VK_TRUE, ~((uint64_t)0)));
	_VK(vkResetFences(qvk.device, 1, &slot->pack_fence));
	slot->fence_pending = true;

	slot->dump = upscaler.dump_requested;
	upscaler.dump_requested = false;

	slot->packed = true;

	return VK_SUCCESS;
}

// The fence the frame's post command buffer must signal, or VK_NULL_HANDLE when
// it carries no pack dispatch -- either pack can bail on a failed staging
// allocation, and the spatial one also on a zero grid or an over-budget tile
// count, without recording one.
VkFence vkpt_upscaler_pack_fence(void)
{
	upscaler_slot_t *slot = spatial_slot_cur();
	return slot->packed ? slot->pack_fence : VK_NULL_HANDLE;
}

// Records the temporal pack pass into `slot`, from vkpt_upscaler_do().
//
// Where the spatial pack is a 1:1 copy into a tile grid, this one builds all
// four model inputs, and its real work is reprojecting the previous frame's
// latent state onto this frame's pixels. That state lives in the *previous*
// slot's output buffer, which ONNX Runtime filled from the CPU moments ago in
// temporal_run_inference() -- host writes made before a submit are visible to
// the device without an explicit barrier, and the buffer is host-coherent.
static VkResult temporal_pack(VkCommandBuffer cmd_buf, upscaler_slot_t *slot)
{
	slot->packed = false;

	if (!ensure_temporal_buffers())
		return VK_SUCCESS;

	upscaler_slot_t *prev = &upscaler.slots[temporal_prev_slot_index(qvk.current_frame_index)];

	// There is history to warp only if the previous inference actually produced
	// a state, and only if it describes the same pixel grid this frame is on.
	// Otherwise the model gets zeros, which is its own "no history" input --
	// meaning a reset costs nothing and reconstructs over the next few frames.
	bool has_history = prev->tensor_valid
		&& prev->lr_width == upscaler.lr_width
		&& prev->lr_height == upscaler.lr_height;

	slot->temporal_push = (upscaler_temporal_push_constants_t){
		.lr_width         = upscaler.lr_width,
		.lr_height        = upscaler.lr_height,
		.out_width        = upscaler.lr_width * upscaler.scale,
		.out_height       = upscaler.lr_height * upscaler.scale,
		.state_channels   = upscaler.state_channels,
		.has_history      = has_history ? 1u : 0u,
		.reproj_threshold = cvar_flt_upscaler_reproj_threshold->value,
	};

	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);
	BEGIN_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_TEMPORAL_PACK);

	bind_upscaler_pipeline(cmd_buf, upscaler.pipeline_pack, slot);

	vkCmdDispatch(cmd_buf,
		(upscaler.lr_width + 7) / 8,
		(upscaler.lr_height + 7) / 8,
		1);

	// Make the shader writes visible to the host read that next frame's
	// temporal_run_inference() will do once pack_fence reports this dispatch done.
	BUFFER_BARRIER(cmd_buf,
		.buffer        = slot->buf_input.buffer,
		.offset        = 0,
		.size          = VK_WHOLE_SIZE,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
	);

	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER_TEMPORAL_PACK);
	END_PERF_MARKER(cmd_buf, PROFILER_UPSCALER);

	// Same fence discipline as the spatial path: it can still be pending if the
	// frame that would have waited on it never ran, and resetting a pending fence
	// is invalid. See spatial_pack().
	if (slot->fence_pending)
		_VK(vkWaitForFences(qvk.device, 1, &slot->pack_fence, VK_TRUE, ~((uint64_t)0)));
	_VK(vkResetFences(qvk.device, 1, &slot->pack_fence));
	slot->fence_pending = true;

	slot->dump = upscaler.dump_requested;
	upscaler.dump_requested = false;

	slot->packed = true;

	return VK_SUCCESS;
}

// Dumps the interesting planes of a temporal tensor pair as PNGs. The confidence
// channel and the warped state are the point: the reprojection is the part of
// this integration the engine owns outright and the model cannot correct for, so
// being able to look at what actually goes in matters more than the colour.
static void temporal_dump_tensors(const upscaler_slot_t *slot)
{
	uint32_t w = upscaler.lr_width, h = upscaler.lr_height;
	uint32_t out_w = w * upscaler.scale, out_h = h * upscaler.scale;

	const float *in  = (const float *)((const uint8_t *)slot->input_mapped);
	const float *out = (const float *)((const uint8_t *)slot->output_mapped
		+ upscaler.temporal_output_offsets[TEMPORAL_OUT_COLOR]);

	size_t plane = (size_t)w * h;
	const float *color      = in + 0;
	const float *state      = in + 3 * plane;
	const float *confidence = in + (3 + upscaler.state_channels) * plane;

	// The two numbers that say whether the temporal path is actually temporal.
	// A zero mean confidence with has_history=1 means the reprojection test is
	// rejecting everything; with has_history=0 it means no state reached the pack
	// at all. The images below cannot tell those apart, and they look identical.
	double confidence_sum = 0.0;
	for (size_t i = 0; i < plane; i++)
		confidence_sum += confidence[i];

	Com_Printf("upscaler: dump frame %" PRIu64 ": has_history %u, mean confidence %.3f\n",
		qvk.frame_counter, slot->temporal_push.has_history, confidence_sum / (double)plane);

	upscaler_dump_float_rgb("pre", color, color + plane, color + 2 * plane, w, h);
	upscaler_dump_float_rgb("post", out, out + (size_t)out_w * out_h, out + 2 * (size_t)out_w * out_h, out_w, out_h);
	upscaler_dump_float_gray("confidence", confidence, w, h);

	// The first few state channels, enough to see whether the warp is producing
	// something spatially coherent without writing sixteen images.
	uint32_t state_dumps = min(upscaler.state_channels, 4u);
	for (uint32_t c = 0; c < state_dumps; c++) {
		char stage[32];
		Q_snprintf(stage, sizeof(stage), "state%u", c);
		upscaler_dump_float_gray(stage, state + (size_t)c * plane, w, h);
	}
}

// Runs the NPU for the tensor `slot` was packed with -- the previous frame's,
// because the round trip is pipelined. One inference for the whole frame, and
// two outputs: the upscaled colour the unpack will present, and the latent state
// the *next* pack will warp forward. That ordering is what makes the pipelining
// free here rather than a compromise: the state comes back before this frame's
// pack is recorded, so the model still sees exactly one frame of history.
static VkResult temporal_run_inference(upscaler_slot_t *slot)
{
	if (!slot->packed)
		return VK_SUCCESS;

	slot->packed = false;
	slot->tensor_valid = false;

	bool dump = slot->dump;
	slot->dump = false;

	unsigned wait_begin = Sys_Milliseconds();

	_VK(vkWaitForFences(qvk.device, 1, &slot->pack_fence, VK_TRUE, ~((uint64_t)0)));
	slot->fence_pending = false;

	unsigned time_begin = Sys_Milliseconds();

	OrtValue *inputs[TEMPORAL_NUM_INPUTS] = { NULL };
	OrtValue *outputs[TEMPORAL_NUM_OUTPUTS] = { NULL };
	bool ok = true;

	// Wrapping the mapped staging directly, so neither side of the inference
	// copies. Note the tensors are float32 even though the weights are int8: the
	// quantize/dequantize pairs live inside the graph.
	for (int i = 0; i < TEMPORAL_NUM_INPUTS && ok; i++) {
		ok = ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info,
			(uint8_t *)slot->input_mapped + upscaler.temporal_input_offsets[i],
			upscaler.temporal_input_bytes[i], upscaler.temporal_input_dims[i], UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[i]), "CreateTensorWithDataAsOrtValue(input)");
	}

	for (int i = 0; i < TEMPORAL_NUM_OUTPUTS && ok; i++) {
		ok = ort_ok(upscaler.api->CreateTensorWithDataAsOrtValue(upscaler.cpu_memory_info,
			(uint8_t *)slot->output_mapped + upscaler.temporal_output_offsets[i],
			upscaler.temporal_output_bytes[i], upscaler.temporal_output_dims[i], UPSCALER_NUM_DIMS,
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputs[i]), "CreateTensorWithDataAsOrtValue(output)");
	}

	if (ok) {
		OrtStatus *status = upscaler.api->Run(upscaler.session, NULL,
			temporal_input_names, (const OrtValue * const *)inputs, TEMPORAL_NUM_INPUTS,
			temporal_output_names, TEMPORAL_NUM_OUTPUTS, outputs);

		if (status) {
			Com_EPrintf("upscaler: temporal inference failed: %s\n", upscaler.api->GetErrorMessage(status));
			upscaler.api->ReleaseStatus(status);
			ok = false;
		}
	}

	for (int i = 0; i < TEMPORAL_NUM_INPUTS; i++)
		if (inputs[i])
			upscaler.api->ReleaseValue(inputs[i]);
	for (int i = 0; i < TEMPORAL_NUM_OUTPUTS; i++)
		if (outputs[i])
			upscaler.api->ReleaseValue(outputs[i]);

	// A failed inference leaves both the colour and the state holding whatever
	// was there before. Dropping the whole slot rather than only the colour is
	// deliberate: feeding a stale state back in would keep the failure circulating
	// through the feedback loop long after the frame that caused it.
	slot->tensor_valid = ok;

	if (dump && ok)
		temporal_dump_tensors(slot);

	upscaler.wait_ms_accum += time_begin - wait_begin;
	upscaler.inference_ms_accum += Sys_Milliseconds() - time_begin;
	upscaler.inference_frames++;

	if (upscaler.inference_frames >= UPSCALER_TIMING_INTERVAL)
	{
		Com_Printf("upscaler: %.2f ms/frame for one %ux%u inference, %.2f ms/frame waiting on the pack\n",
			(double)upscaler.inference_ms_accum / upscaler.inference_frames,
			upscaler.lr_width, upscaler.lr_height,
			(double)upscaler.wait_ms_accum / upscaler.inference_frames);
		upscaler.inference_ms_accum = 0;
		upscaler.wait_ms_accum = 0;
		upscaler.inference_frames = 0;
	}

	return VK_SUCCESS;
}

// Runs the NPU for the tensor `slot` was packed with, which -- because the round
// trip is pipelined -- is the *previous* frame's. No queue drain: the pack
// dispatch's own submit signalled pack_fence, and a whole frame of GPU work has
// happened since, so this wait is usually already satisfied. Where it is not,
// it blocks only on the previous frame's post pass rather than on everything
// submitted so far, and the GPU stays free to work through the current frame.
static VkResult spatial_run_inference(upscaler_slot_t *slot)
{
	if (!slot->packed)
		return VK_SUCCESS;

	slot->packed = false;
	slot->tensor_valid = false;

	bool dump = slot->dump;
	slot->dump = false;

	unsigned wait_begin = Sys_Milliseconds();

	_VK(vkWaitForFences(qvk.device, 1, &slot->pack_fence, VK_TRUE, ~((uint64_t)0)));
	slot->fence_pending = false;

	unsigned time_begin = Sys_Milliseconds();

	uint32_t num_tiles = slot->tiles_x * slot->tiles_y;
	const char *input_names[]  = { UPSCALER_INPUT_NAME };
	const char *output_names[] = { UPSCALER_OUTPUT_NAME };
	bool all_tiles_ok = true;

	for (uint32_t tile = 0; tile < num_tiles; tile++)
	{
		uint8_t *in  = (uint8_t *)slot->input_mapped  + (size_t)tile * upscaler.input_byte_size;
		uint8_t *out = (uint8_t *)slot->output_mapped + (size_t)tile * upscaler.output_byte_size;

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
	slot->tensor_valid = all_tiles_ok;

	upscaler.wait_ms_accum += time_begin - wait_begin;
	upscaler.inference_ms_accum += Sys_Milliseconds() - time_begin;
	upscaler.inference_frames++;

	if (upscaler.inference_frames >= UPSCALER_TIMING_INTERVAL)
	{
		Com_Printf("upscaler: %.2f ms/frame for %u tiles (%.2f ms/tile), %.2f ms/frame waiting on the pack\n",
			(double)upscaler.inference_ms_accum / upscaler.inference_frames,
			num_tiles,
			(double)upscaler.inference_ms_accum / (upscaler.inference_frames * num_tiles),
			(double)upscaler.wait_ms_accum / upscaler.inference_frames);
		upscaler.inference_ms_accum = 0;
		upscaler.wait_ms_accum = 0;
		upscaler.inference_frames = 0;
	}

	return VK_SUCCESS;
}

// The frame's NPU work: the inference for whatever the previous frame packed,
// then this frame's pack. Running the inference here, before anything is
// recorded, gives pack_fence the whole preceding frame to be signalled in.
VkResult vkpt_upscaler_do(VkCommandBuffer cmd_buf)
{
	if (temporal_is_active()) {
		// Order matters more here than it does for the spatial path: the state
		// this inference returns is what the pack below warps into the frame.
		temporal_run_inference(spatial_slot_prev());
		return temporal_pack(cmd_buf, spatial_slot_cur());
	}

	if (!spatial_is_active())
		return VK_SUCCESS;

	spatial_run_inference(spatial_slot_prev());

	return spatial_pack(cmd_buf, spatial_slot_cur());
}

VkResult vkpt_upscaler_final_blit(VkCommandBuffer cmd_buf, bool warp)
{
	// One frame behind: the tensor this unpacks was packed last frame and
	// inferred at the top of this one. A slot's result is good for exactly one
	// blit, so take it and clear it.
	upscaler_slot_t *slot = spatial_slot_prev();
	bool tensor_valid = slot->tensor_valid;
	slot->tensor_valid = false;

	// The pass can bail before producing anything -- a failed staging allocation
	// leaves the buffers destroyed, a failed inference leaves the tensor holding
	// an older frame, and on a cold start there is no previous frame at all.
	// Unpacking any of those would present garbage or read an already-freed
	// buffer, so fall back to the tone-mapped frame, which is below display
	// resolution here, hence the filtered blit.
	if (!tensor_valid) {
		bool needs_filter = qvk.extent_taa_output.width  != qvk.extent_unscaled.width
		                 || qvk.extent_taa_output.height != qvk.extent_unscaled.height;
		return vkpt_final_blit(cmd_buf, VKPT_IMG_TAA_OUTPUT, qvk.extent_taa_output,
			needs_filter, warp);
	}

	bool temporal = upscaler.kind == UPSCALER_KIND_TEMPORAL;
	int marker = temporal ? PROFILER_UPSCALER_TEMPORAL_UNPACK : PROFILER_UPSCALER_UNPACK;

	BEGIN_PERF_MARKER(cmd_buf, marker);

	bind_upscaler_pipeline(cmd_buf, upscaler.pipeline_unpack, slot);

	// Unpack writes only the display rect of IMG_UPSCALE_OUTPUT, matching what
	// the final blit below samples back out of it.
	VkExtent2D out = qvk.extent_unscaled;
	vkCmdDispatch(cmd_buf, (out.width + 7) / 8, (out.height + 7) / 8, 1);

	BARRIER_COMPUTE(cmd_buf, qvk.images[VKPT_IMG_UPSCALE_OUTPUT]);

	END_PERF_MARKER(cmd_buf, marker);

	return vkpt_final_blit(cmd_buf, VKPT_IMG_UPSCALE_OUTPUT, qvk.extent_unscaled, false, warp);
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

VkFence vkpt_upscaler_pack_fence(void)
{
	return VK_NULL_HANDLE;
}

void vkpt_upscaler_discard(void)
{
}

bool vkpt_upscaler_wants_input_tap(void)
{
	return false;
}

bool vkpt_upscaler_get_temporal_extent(VkExtent2D *extent)
{
	return false;
}

void vkpt_upscaler_check_render_extent(void)
{
}

#endif // USE_ORT_QNN_UPSCALER

// vim: shiftwidth=4 noexpandtab tabstop=4 cindent
