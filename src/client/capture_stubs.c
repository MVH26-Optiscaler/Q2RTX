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
	Stubs for the q2rtx_capture build
	=================================

	The capture tool (client/capture_bot.c) drives the game with nobody watching
	or listening, so it is built without the sound engine, the menus, the
	cinematic player, demo recording, the GTV client and the HTTP downloader --
	see SRC_CAPTURE in src/CMakeLists.txt.

	The rest of the client calls into all of those unconditionally, so rather
	than sprinkling #ifs through a dozen files this provides the small set of
	symbols that go missing. Everything here does nothing; the two exceptions
	are noted where they are.

	This file is compiled into the capture tool only. The normal game keeps the
	real implementations.
*/

#include "client.h"
#include "ui/ui.h"

/*
================================================================================
Sound
================================================================================
*/

// client/entities.c updates the listener every frame regardless of whether
// anything is going to read it.
sndstarted_t s_started;
vec3_t       listener_origin;
vec3_t       listener_forward;
vec3_t       listener_right;
vec3_t       listener_up;
int          listener_entnum;

void S_Init(void) { }
void S_Shutdown(void) { }
void S_Activate(void) { }
void S_Update(void) { }
void S_ParseStartSound(void) { }
void S_FreeAllSounds(void) { }
void S_StopAllSounds(void) { }
void S_BeginRegistration(void) { }
void S_EndRegistration(void) { }
void S_StartLocalSound(const char *s) { }
void S_StartLocalSoundOnce(const char *s) { }

void S_StartSound(const vec3_t origin, int entnum, int entchannel,
                  qhandle_t sfx, float fvol, float attenuation, float timeofs) { }

// 0 is the "no such sound" handle, which every caller already has to handle for
// sounds a mod failed to ship.
qhandle_t S_RegisterSound(const char *sample) { return 0; }

void OGG_Init(void) { }
void OGG_Shutdown(void) { }
void OGG_Play(void) { }
void OGG_Stop(void) { }
void OGG_LoadTrackList(void) { }

/*
================================================================================
Menus
================================================================================
*/

// refresh/vkpt/main.c reads uis.menuDepth every frame to decide whether to blur
// the view for the menu. Left zeroed, it never is.
uiStatic_t uis;

void UI_Init(void) { }
void UI_Shutdown(void) { }
void UI_ModeChanged(void) { }
void UI_Frame(int msec) { }
void UI_Draw(unsigned realtime) { }
void UI_OpenMenu(uiMenu_t menu) { }
void UI_KeyEvent(int key, bool down) { }
void UI_CharEvent(int key) { }
void UI_MouseEvent(int x, int y) { }
void UI_StatusEvent(const serverStatus_t *status) { }
void UI_ErrorEvent(netadr_t *from) { }

// Says "the menu is not covering the view", which is what the screen code needs
// to hear to keep drawing the world.
bool UI_IsTransparent(void) { return true; }

/*
================================================================================
Cinematics, demos, GTV

The HTTP downloader needs nothing here: client.h already reduces the whole
HTTP_* API to no-op macros when USE_CURL is off, which is how this target is
built.
================================================================================
*/

void SCR_PlayCinematic(const char *name) { }
void SCR_StopCinematic(void) { }
void SCR_FinishCinematic(void) { }
void SCR_RunCinematic(void) { }
void SCR_ReloadCinematic(void) { }
void SCR_DrawCinematic(void) { }

void CL_InitDemos(void) { }
void CL_CleanupDemos(void) { }
void CL_DemoFrame(int msec) { }
void CL_EmitDemoFrame(void) { }
void CL_FirstDemoFrame(void) { }
void CL_FreeDemoSnapshots(void) { }
void CL_Stop_f(void) { }
bool CL_WriteDemoMessage(sizebuf_t *buf) { return false; }

void CL_GTV_Init(void) { }
void CL_GTV_Shutdown(void) { }
void CL_GTV_Run(void) { }
void CL_GTV_Resume(void) { }
void CL_GTV_Suspend(void) { }
void CL_GTV_Transmit(void) { }
void CL_GTV_EmitFrame(void) { }
void CL_GTV_WriteMessage(byte *data, size_t len) { }
