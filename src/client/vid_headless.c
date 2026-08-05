/*
Copyright (C) 2013 Andrey Nazarov
Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.

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

//
// vid_headless.c - windowless video driver
//
// Selected with "+set vid_driver headless" on the command line, which is early
// enough to beat CL_InitRefresh(). Never probed, so it is opt-in only. There is
// no window, no display connection and no event source: the renderer creates a
// VK_EXT_headless_surface instead of an SDL one, and the render resolution is
// whatever vid_geometry says rather than whatever a compositor hands out. Used
// by the dataset capture tool, which reads its frames back from an offscreen
// image anyway (refresh/vkpt/capture.c).
//

#include "shared/shared.h"
#include "common/cvar.h"
#include "common/common.h"
#include "common/zone.h"
#include "client/client.h"
#include "client/video.h"
#include "refresh/refresh.h"

static struct {
    int     width;
    int     height;
    bool    activated;
} headless;

static void mode_changed(void)
{
    vrect_t rc;

    VID_GetGeometry(&rc);

    headless.width = rc.width;
    headless.height = rc.height;

    R_ModeChanged(headless.width, headless.height, 0);
    SCR_ModeChanged();
}

static bool init(graphics_api_t api)
{
    if (api != GAPI_VULKAN) {
        Com_SetLastError("headless video driver requires the Vulkan renderer");
        return false;
    }

    mode_changed();

    // Nothing will ever deliver a window event here, and cls.active is
    // zero-initialized to ACT_MINIMIZED, which pins CL_UpdateFrameTimes() to
    // SYNC_SLEEP_10 and stops the client from rendering at all.
    CL_Activate(ACT_ACTIVATED);
    headless.activated = true;

    Com_Printf("Headless video driver initialized (%dx%d).\n",
               headless.width, headless.height);
    return true;
}

static void headless_shutdown(void)
{
    memset(&headless, 0, sizeof(headless));
}

static void fatal_shutdown(void)
{
}

static void pump_events(void)
{
}

static char *get_mode_list(void)
{
    return Z_CopyString(va("%dx%d", headless.width, headless.height));
}

static int get_dpi_scale(void)
{
    return 1;
}

static void set_mode(void)
{
    mode_changed();
}

static void update_gamma(const byte *table)
{
}

static char *get_selection_data(void)
{
    return NULL;
}

static char *get_clipboard_data(void)
{
    return NULL;
}

static void set_clipboard_data(const char *data)
{
}

static bool init_mouse(void)
{
    return false;
}

static void shutdown_mouse(void)
{
}

static bool get_mouse_motion(int *dx, int *dy)
{
    return false;
}

static bool probe(void)
{
    // opt-in only: never picked by the fallback loop in CL_InitRefresh()
    return false;
}

const vid_driver_t vid_headless = {
    .name = "headless",
    .headless = true,

    .probe = probe,
    .init = init,
    .shutdown = headless_shutdown,
    .fatal_shutdown = fatal_shutdown,
    .pump_events = pump_events,

    .get_mode_list = get_mode_list,
    .get_dpi_scale = get_dpi_scale,
    .set_mode = set_mode,
    .update_gamma = update_gamma,

    .get_selection_data = get_selection_data,
    .get_clipboard_data = get_clipboard_data,
    .set_clipboard_data = set_clipboard_data,

    // grab_mouse and warp_mouse stay NULL: IN_Activate() and IN_WarpMouse()
    // skip them when they are, and IN_Activate() would otherwise walk into
    // IN_GetCurrentGrab() reading the in_grab cvar, which IN_Init() never
    // creates once init_mouse() has failed.
    .init_mouse = init_mouse,
    .shutdown_mouse = shutdown_mouse,
    .get_mouse_motion = get_mouse_motion,
};
