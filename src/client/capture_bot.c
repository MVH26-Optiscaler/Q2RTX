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
	Training-data capture bot
	=========================

	Walks a level -- or every level in the game -- unattended, so that the
	renderer's capture pass (refresh/vkpt/capture.c) can collect ground-truth
	tiles for training the NPU upscaler without a human flying the camera around
	forty maps.

	It moves the real player entity, through the 'setpos' cheat added to the game
	library, rather than moving the render camera on its own: the server decides
	which entities and lights to send from the player's position, so a camera
	that wandered off by itself would be photographing rooms with half their
	contents missing.

	The cycle per sample point is: pick a random spot in an empty BSP leaf,
	teleport there, then look in several random directions, holding still for a
	while before each shot. Holding still is the point of the design -- a
	path-traced frame in motion is still converging and shows ASVGF and TAA
	trails, and tiles full of those would teach the model to reproduce them.

	Sample points come from the BSP leaf array, which is also what makes them
	trustworthy: a leaf with a cluster of -1 is the sealed void outside the
	playable space and renders as nothing at all, so requiring cluster >= 0 and
	empty contents is exactly the test for "this point is inside the map".

	Console commands
	----------------
	* capture_scan [map ...] - scan the listed maps, or every map the filesystem
	  can see if none are given. Loads each one in turn and moves on by itself.
	* capture_scan_stop - stop, and report how far it got.
*/

#include "client.h"

// The teleport is a server round trip, so the client does not see the new
// position until a server frame later -- 100 ms at the stock 10 Hz, which is
// many rendered frames or very few depending on the map and resolution. Hence a
// wall-clock deadline rather than a frame count: still being somewhere else
// when it expires means the move did not happen.
#define BOT_ARRIVE_DIST      96.f
#define BOT_ARRIVE_TIMEOUT   1500
#define BOT_LEAF_TRIES       64     // random leaf picks before giving up
#define BOT_PICK_FAILURES    8      // failed teleports in a row before stopping
#define BOT_CLEARANCE        24.f   // half-extent of the box a sample point must fit
// Frames to hold the pose after issuing a shot. The command is executed at the
// top of one of the next frames and serviced by the frame after that, so the
// camera must not swing out from under it immediately.
#define BOT_SHOOT_FRAMES     3
// Milliseconds to wait for a level to come up before writing it off.
#define BOT_MAP_TIMEOUT      60000

typedef enum {
	BOT_IDLE,
	BOT_WAIT_MAP,  // map is loading / not active yet
	BOT_PICK,      // choose the next sample point and teleport to it
	BOT_ARRIVE,    // wait for the server to acknowledge the teleport
	BOT_SETTLE,    // hold still while the renderer converges
	BOT_SHOOT,     // shot issued, wait for it to be serviced
} bot_state_t;

static cvar_t *cvar_capture_looks;
static cvar_t *cvar_capture_settle;
static cvar_t *cvar_capture_samples;
static cvar_t *cvar_capture_tag;

static struct {
	bot_state_t state;

	// Maps still to visit. Always an FS_ListFiles()-shaped list, whether it came
	// from the filesystem or was built from command arguments, so both are freed
	// the same way.
	void      **maps;
	int         num_maps;
	int         next_map;
	// The map we asked for. 'map' is queued, not immediate, so cls.state is
	// still ca_active on the old level for several frames afterwards; without
	// this the bot would happily start scanning the level it is leaving.
	char        pending_map[MAX_QPATH];
	unsigned    pending_since;  // cls.realtime when the map was asked for

	vec3_t      destination;
	int         failed_picks;

	int         look;          // index into this stop's look directions
	int         wait_frames;   // rendered frames left to hold still

	int         shots_map;     // shots taken on the current map
	int         shots_total;
	int         maps_done;

	// The angles CL_CaptureBot_UpdateCmd() forces onto the outgoing usercmd.
	// Chosen by the state machine, not by the cmd hook, so they stay put across
	// the several usercmds built between two rendered frames.
	vec3_t      want_angles;
} bot;

static float bot_frand_range(float lo, float hi)
{
	return lo + frand() * (hi - lo);
}

/*
=================
bot_pick_destination

Uniformly samples empty BSP leaves rather than the map bounding box: most of a
Quake 2 level's bounding box is solid, so rejection sampling a box would spend
most of its attempts inside walls.
=================
*/
static bool bot_pick_destination(vec3_t out)
{
	if (!cl.bsp || cl.bsp->numleafs <= 0)
		return false;

	for (int i = 0; i < BOT_LEAF_TRIES; i++) {
		const mleaf_t *leaf = &cl.bsp->leafs[Q_rand() % cl.bsp->numleafs];

		if (leaf->cluster < 0 || leaf->contents)
			continue;

		vec3_t point;
		bool fits = true;

		for (int axis = 0; axis < 3; axis++) {
			float lo = leaf->mins[axis] + BOT_CLEARANCE;
			float hi = leaf->maxs[axis] - BOT_CLEARANCE;

			if (lo > hi) {
				// Leaf is thinner than the camera needs; a sample taken here
				// would be half-buried in a wall.
				fits = false;
				break;
			}

			point[axis] = bot_frand_range(lo, hi);
		}

		if (!fits)
			continue;

		if (CM_PointContents(point, cl.bsp->nodes) & MASK_SOLID)
			continue;

		// The leaf bounds are loose, so confirm the box around the point is
		// actually clear before committing to it.
		trace_t trace;
		const vec3_t mins = { -BOT_CLEARANCE, -BOT_CLEARANCE, -BOT_CLEARANCE };
		const vec3_t maxs = {  BOT_CLEARANCE,  BOT_CLEARANCE,  BOT_CLEARANCE };

		CM_BoxTrace(&trace, point, point, mins, maxs, cl.bsp->nodes, MASK_SOLID);
		if (trace.startsolid || trace.allsolid)
			continue;

		VectorCopy(point, out);
		return true;
	}

	return false;
}

// True when the camera is somewhere a capture is worth taking: inside the map,
// not buried in geometry.
static bool bot_view_is_valid(void)
{
	if (!cl.bsp)
		return false;

	if (CM_PointContents(cl.refdef.vieworg, cl.bsp->nodes) & MASK_SOLID)
		return false;

	const mleaf_t *leaf = BSP_PointLeaf(cl.bsp->nodes, cl.refdef.vieworg);
	return leaf && leaf->cluster >= 0;
}

static void bot_free_maps(void)
{
	if (bot.maps) {
		FS_FreeList(bot.maps);
		bot.maps = NULL;
	}

	bot.num_maps = 0;
	bot.next_map = 0;
}

static void bot_stop(const char *reason)
{
	if (bot.state == BOT_IDLE)
		return;

	bot.state = BOT_IDLE;
	bot_free_maps();

	Com_Printf("capture bot: stopped (%s) after %d shots across %d maps\n",
		reason, bot.shots_total, bot.maps_done);

	// This build exists to run exactly one pass, so there is nothing left to do
	// once the pass is over.
	Cbuf_AddText(&cmd_buffer, "quit\n");
}

// Cheats needed to stand anywhere in the level and be ignored while doing it.
// Issued after every level change, because they do not survive one. 'noclip' is
// what keeps the player from falling out of a mid-air sample point.
static void bot_enter_map(void)
{
	Cbuf_AddText(&cmd_buffer, "god\nnotarget\nnoclip\n");
	Cvar_SetByVar(cvar_capture_tag, cl.mapname, FROM_CODE);

	bot.shots_map = 0;
	bot.failed_picks = 0;
	bot.state = BOT_PICK;

	Com_Printf("capture bot: scanning %s (%d of %d)\n",
		cl.mapname, bot.next_map, bot.num_maps);
}

static void bot_next_map(void)
{
	if (bot.next_map >= bot.num_maps) {
		bot_stop("finished the map list");
		return;
	}

	const char *map = bot.maps[bot.next_map++];

	Com_Printf("capture bot: loading %s\n", map);
	Q_strlcpy(bot.pending_map, map, sizeof(bot.pending_map));
	bot.pending_since = cls.realtime;
	Cbuf_AddText(&cmd_buffer, va("map %s\n", map));

	bot.state = BOT_WAIT_MAP;
}

// Aims somewhere new for the current stop. The pitch range is deliberately
// lopsided: floors carry more texture detail than ceilings, and straight up is
// usually a flat lightmapped slab.
static void bot_pick_look(void)
{
	bot.want_angles[PITCH] = bot_frand_range(-25.f, 15.f);
	bot.want_angles[YAW]   = bot_frand_range(0.f, 360.f);
	bot.want_angles[ROLL]  = 0.f;

	bot.wait_frames = max(1, cvar_capture_settle->integer);
}

// Counts down a hold that is measured in rendered frames. Returns true when it
// has expired, false while there is still waiting to do.
static bool bot_hold_expired(bool rendered)
{
	if (!rendered)
		return false;

	return --bot.wait_frames <= 0;
}

/*
=================
CL_CaptureBot_Frame

Runs once per client frame; 'rendered' says whether a frame was actually drawn,
which is what the settle countdown has to count -- the renderer converges per
rendered frame, not per client frame. Called after the render, so the state
decided here applies from the next frame onward.
=================
*/
void CL_CaptureBot_Frame(bool rendered)
{
	if (bot.state == BOT_IDLE)
		return;

	if (cls.state != ca_active) {
		// Either a level is loading or the connection dropped; either way there
		// is nothing to drive yet.
		bot.state = BOT_WAIT_MAP;
		return;
	}

	if (bot.state == BOT_WAIT_MAP) {
		// Wait for the level we actually asked for, not just for any level to
		// be running.
		if (Q_stricmp(cl.mapname, bot.pending_map)) {
			// A map the server refuses to load would otherwise stall the whole
			// pass, and nobody is watching to notice.
			if (cls.realtime - bot.pending_since > BOT_MAP_TIMEOUT) {
				Com_WPrintf("capture bot: %s did not come up, skipping it\n", bot.pending_map);
				bot_next_map();
			}
			return;
		}

		bot_enter_map();
		return;
	}

	switch (bot.state) {
	case BOT_PICK:
		if (!bot_pick_destination(bot.destination)) {
			bot_stop("could not find anywhere to stand");
			return;
		}

		Cbuf_AddText(&cmd_buffer, va("cmd setpos %.1f %.1f %.1f\n",
			bot.destination[0], bot.destination[1], bot.destination[2]));

		bot.pending_since = cls.realtime;
		bot.state = BOT_ARRIVE;
		break;

	case BOT_ARRIVE: {
		vec3_t delta;
		VectorSubtract(bot.destination, cl.predicted_origin, delta);

		if (VectorLength(delta) > BOT_ARRIVE_DIST) {
			// Keep watching until the deadline; the move lands on whichever
			// frame the next server update happens to arrive on.
			if (cls.realtime - bot.pending_since <= BOT_ARRIVE_TIMEOUT)
				break;

			// The teleport did not land: either the server refused the command
			// or the point was rejected. Try somewhere else, but do not spin on
			// it forever if 'setpos' is simply not there.
			if (++bot.failed_picks >= BOT_PICK_FAILURES) {
				bot_stop("teleports are not landing -- is this build's game library current?");
				return;
			}

			bot.state = BOT_PICK;
			break;
		}

		bot.failed_picks = 0;
		bot.look = 0;
		bot_pick_look();
		bot.state = BOT_SETTLE;
		break;
	}

	case BOT_SETTLE:
		if (!bot_hold_expired(rendered))
			break;

		if (bot_view_is_valid()) {
			Cbuf_AddText(&cmd_buffer, "capture_shot\n");
			bot.shots_map++;
			bot.shots_total++;
		}

		bot.wait_frames = BOT_SHOOT_FRAMES;
		bot.state = BOT_SHOOT;
		break;

	case BOT_SHOOT:
		if (!bot_hold_expired(rendered))
			break;

		if (bot.shots_map >= cvar_capture_samples->integer) {
			bot.maps_done++;
			bot_next_map();
			break;
		}

		if (++bot.look < cvar_capture_looks->integer) {
			bot_pick_look();
			bot.state = BOT_SETTLE;
		} else {
			bot.state = BOT_PICK;
		}
		break;

	default:
		break;
	}
}

/*
=================
CL_CaptureBot_UpdateCmd

Overrides the human's input with the bot's. Called from CL_UpdateCmd after the
keyboard and mouse have had their say, so whatever they contributed is simply
discarded while the bot is driving. Movement is always zero: the bot teleports
rather than walking, and any residual movement would drift the camera out of the
pose the capture was taken at.
=================
*/
void CL_CaptureBot_UpdateCmd(void)
{
	if (bot.state == BOT_IDLE || bot.state == BOT_WAIT_MAP)
		return;

	VectorCopy(bot.want_angles, cl.viewangles);
	VectorClear(cl.localmove);
}

bool CL_CaptureBot_Active(void)
{
	return bot.state != BOT_IDLE;
}

// Turns command arguments into the same shape FS_ListFiles() returns, so that
// an explicit map list and a scanned one are freed and indexed identically.
static void **bot_list_from_args(int *count)
{
	int num = Cmd_Argc() - 1;
	void **list = FS_ReallocList(NULL, num + 1);

	for (int i = 0; i < num; i++)
		list[i] = Z_CopyString(Cmd_Argv(i + 1));

	list[num] = NULL;
	*count = num;
	return list;
}

static void CL_CaptureScan_f(void)
{
	memset(&bot, 0, sizeof(bot));

	if (Cmd_Argc() > 1) {
		bot.maps = bot_list_from_args(&bot.num_maps);
	} else {
		// Same flags SV_Map_c() completes 'map' with, so the names come back
		// directly usable as map arguments -- including the ones inside pak0.pak.
		bot.maps = FS_ListFiles("maps", "*.bsp",
			FS_SEARCH_SAVEPATH | FS_SEARCH_BYFILTER | FS_SEARCH_STRIPEXT, &bot.num_maps);
	}

	if (!bot.maps || bot.num_maps <= 0) {
		Com_Printf("capture bot: no maps to scan\n");
		bot_free_maps();
		return;
	}

	Com_Printf("capture bot: scanning %d maps, %d shots each\n",
		bot.num_maps, cvar_capture_samples->integer);

	bot.state = BOT_WAIT_MAP;
	bot_next_map();
}

// What the startup sequence runs. Scanning everything is the default, but a
// capture_user.cfg that already started a scan of its own -- with a map list,
// say -- wins, because it is read first.
static void CL_CaptureScanAuto_f(void)
{
	if (CL_CaptureBot_Active())
		return;

	CL_CaptureScan_f();
}

static void CL_CaptureScanStop_f(void)
{
	bot_stop("asked to");
}

static const cmdreg_t c_capture_bot[] = {
	{ "capture_scan", CL_CaptureScan_f },
	{ "capture_scan_auto", CL_CaptureScanAuto_f },
	{ "capture_scan_stop", CL_CaptureScanStop_f },
	{ NULL, NULL }
};

void CL_CaptureBot_Init(void)
{
	// How many directions to look from each stop, and how many rendered frames
	// to hold each pose. The default hold is generous because the path tracer
	// and TAA need a good dozen frames to settle after the camera stops, and a
	// half-converged tile is worse than no tile.
	cvar_capture_looks = Cvar_Get("capture_looks", "4", CVAR_ARCHIVE);
	cvar_capture_settle = Cvar_Get("capture_settle", "16", CVAR_ARCHIVE);
	// Shots per map before moving on. Each shot writes up to
	// capture_tiles_per_frame tiles, so this is not the tile count.
	cvar_capture_samples = Cvar_Get("capture_samples", "25", CVAR_ARCHIVE);
	cvar_capture_tag = Cvar_Get("capture_tag", "capture", 0);

	Cmd_Register(c_capture_bot);
}
