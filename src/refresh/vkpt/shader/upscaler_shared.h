/*
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

// Color transfer shared between upscaler.c and the pack/unpack shaders.
//
// The tile geometry deliberately does not live here: upscaler.c queries it off
// whichever model is loaded and pushes it to the shaders, so a model with a
// different input/output shape is a data change rather than a code change.

#ifndef UPSCALER_SHARED_H_
#define UPSCALER_SHARED_H_

#ifdef VKPT_SHADER

// The upscaler operates on the tone-mapped, display-referred TAA output.
// Encode/decode are inverses of each other; exact curve matters less than the
// round trip being symmetric, since the model just needs plausible sRGB input.
vec3 upscaler_linear_to_encoded(vec3 color)
{
	return pow(clamp(color, vec3(0), vec3(1)), vec3(1.0 / 2.2));
}

vec3 upscaler_encoded_to_linear(vec3 color)
{
	return pow(clamp(color, vec3(0), vec3(1)), vec3(2.2));
}

#endif // VKPT_SHADER

#endif // UPSCALER_SHARED_H_
