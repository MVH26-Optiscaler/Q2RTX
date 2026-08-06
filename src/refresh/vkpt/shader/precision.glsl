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

#ifndef _PRECISION_GLSL_
#define _PRECISION_GLSL_

// Relaxed-precision opt-in, for Adreno.
//
// Shaders are compiled with --target-env vulkan1.2, which makes glslang's
// spvVersion.vulkan > 0, which makes it respect precision qualifiers even in
// the desktop #version 450 profile (ParseHelper.cpp, respectPrecisionQualifiers).
// EpqMedium then lowers to spv::DecorationRelaxedPrecision, which Adreno
// executes at fp16 rate with half the register pressure.
//
// Only quantities that are provably bounded within fp16 range are annotated.
// fp16 is: max 65504, min normal 6.1e-5, ~3.3 decimal digits. In practice that
// means the annotated set is limited to values that are already stored at 16
// bits or less (rgba16f / rg16f / r16f / rg8 / rgba8 images, packHalf2x16
// fields), values clamped to [0,1], and normalized directions. World-space
// positions, absolute screen coordinates, PDFs, GGX denominators, log-space
// exposure, RNG state, and anything reinterpreted via floatBitsToUint are
// deliberately left at full precision -- see doc/mediump-audit.md.
//
// Note on failure modes: clamp_output() in path_tracer_rgen.h maps inf/NaN to
// vec3(0), so a precision regression here shows up as *black* pixels or stale
// history, not as bright fireflies. Diff images, don't eyeball.
//
// To back the whole thing out, compile with -DVKPT_DISABLE_MEDIUMP; every
// annotated declaration reverts to default (highp) precision. Nothing else in
// the tree needs to change.

#ifdef VKPT_DISABLE_MEDIUMP
#define MP
#else
#define MP mediump
#endif

#endif /*_PRECISION_GLSL_*/
