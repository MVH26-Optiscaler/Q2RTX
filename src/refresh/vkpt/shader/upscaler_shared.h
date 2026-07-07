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

// Tensor geometry shared between upscaler.c and the pack/unpack shaders.
// These must match the model's fixed input/output shapes, which upscaler.c
// validates against the loaded ONNX file at startup.

#ifndef UPSCALER_SHARED_H_
#define UPSCALER_SHARED_H_

#define UPSCALER_TILE_IN   128u // model input  is 1x3x128x128 uint8 NCHW
#define UPSCALER_TILE_OUT  512u // model output is 1x3x512x512 uint8 NCHW
#define UPSCALER_SCALE     4u   // UPSCALER_TILE_OUT / UPSCALER_TILE_IN

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
