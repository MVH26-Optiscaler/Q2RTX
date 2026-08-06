set(SHADER_SOURCE_DEPENDENCIES
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/asvgf.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/brdf.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/constants.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/fsr_easu.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/fsr_rcas.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/fsr_utils.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/global_textures.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/global_ubo.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/god_rays_shared.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/light_lists.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/path_tracer_rgen.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/path_tracer.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/path_tracer_hit_shaders.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/path_tracer_transparency.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/precision.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/precomputed_sky.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/precomputed_sky_params.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/projection.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/sky.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/tiny_encryption_algorithm.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/tone_mapping_utils.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/utils.glsl
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/vertex_buffer.h
    ${CMAKE_SOURCE_DIR}/src/refresh/vkpt/shader/water.glsl)

if(TARGET glslang-standalone)
    set(GLSLANG_COMPILER "$<TARGET_FILE:glslang-standalone>")
    message(STATUS "Using glslang built from source")
else()
    find_program(GLSLANG_COMPILER NAMES glslang glslangValidator PATHS "$ENV{VULKAN_SDK}/bin/")

    if(NOT GLSLANG_COMPILER)
        message(FATAL_ERROR "Couldn't find glslang! "
            "Please provide a valid path to it using the GLSLANG_COMPILER variable.")
    endif()
    
    message(STATUS "Using this glslang: ${GLSLANG_COMPILER}")
endif()

# spirv-opt, used to shrink the ray-query compute variants of the path tracer
# shaders (see the OPTIMIZE option of compile_shader below). The bundled
# glslang is built with ENABLE_OPT=OFF -- it advertises -Os but errors out with
# "optimizer not linked" -- so we shell out to a standalone spirv-opt instead.
OPTION(USE_SPIRV_OPT "Run spirv-opt -Os on shaders that request it (needed for ray-query on Adreno)" ON)

if(USE_SPIRV_OPT)
    # VULKAN_SDK isn't always exported, so also probe the default install root.
    file(GLOB VULKAN_SDK_BIN_DIRS "C:/VulkanSDK/*/Bin")
    find_program(SPIRV_OPT_COMMAND NAMES spirv-opt
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" ${VULKAN_SDK_BIN_DIRS})

    if(SPIRV_OPT_COMMAND)
        message(STATUS "Using this spirv-opt: ${SPIRV_OPT_COMMAND}")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
        # Adreno's compute-shader compiler rejects the unoptimized path tracer
        # shaders outright (vkCreateComputePipelines -> VK_ERROR_UNKNOWN), so on
        # ARM64 this isn't merely an optimization -- the build won't run.
        message(WARNING "spirv-opt not found. On Adreno/ARM64 the unoptimized ray-query "
            "path tracer shaders exceed the driver's shader compiler limits and the game "
            "will fail with 'Couldn't initialize pt'. Install the Vulkan SDK or set "
            "SPIRV_OPT_COMMAND.")
    else()
        message(STATUS "spirv-opt not found, shaders will not be size-optimized.")
    endif()
endif()

# Collect additional glslangValidator args
set(GLSLANG_ARGS)
if(CONFIG_BUILD_SHADER_DEBUG_INFO)
    list(APPEND GLSLANG_ARGS -gVS)
endif()

# Write args to a file. Used to trigger rebuild if they change
set(COMPILE_ARGS_DEP "${CMAKE_BINARY_DIR}/compile_shader.dep")
file(CONFIGURE OUTPUT "${COMPILE_ARGS_DEP}" CONTENT "@GLSLANG_ARGS@:@SPIRV_OPT_COMMAND@")

function(compile_shader)
    set(options OPTIMIZE)
    set(oneValueArgs SOURCE_FILE OUTPUT_FILE_NAME OUTPUT_FILE_LIST STAGE)
    set(multiValueArgs DEFINES INCLUDES)
    cmake_parse_arguments(params "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT params_SOURCE_FILE)
        message(FATAL_ERROR "compile_shader: SOURCE_FILE argument missing")
    endif()

    if (NOT params_OUTPUT_FILE_LIST)
        message(FATAL_ERROR "compile_shader: OUTPUT_FILE_LIST argument missing")
    endif()

    set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/${params_SOURCE_FILE}")

    if (params_OUTPUT_FILE_NAME)
        set(output_file_name ${params_OUTPUT_FILE_NAME})
    else()
        get_filename_component(output_file_name ${src_file} NAME)
    endif()

    if (params_STAGE)
        set(stage -S comp)
    else()
        set(stage)
    endif()
    
    set_source_files_properties(${src_file} PROPERTIES VS_TOOL_OVERRIDE "None")

    set (out_dir "${CMAKE_SOURCE_DIR}/baseq2/shader_vkpt")
    set (out_file "${out_dir}/${output_file_name}.spv")
    
    set(glslang_command_line
            ${stage}
            --target-env vulkan1.2
            --quiet
            -DVKPT_SHADER
            -V
            ${GLSLANG_ARGS}
            ${params_DEFINES}
            ${params_INCLUDES}
            "${src_file}"
            -o "${out_file}")

    # Optional in-place size optimization. Shrinks the path tracer's ray-query
    # compute variants by 35-55%, which is what keeps them under the Adreno
    # shader compiler's (undocumented) size limit -- without it those pipelines
    # fail to create with VK_ERROR_UNKNOWN. Harmless elsewhere.
    set(optimize_command)
    if (params_OPTIMIZE AND SPIRV_OPT_COMMAND)
        set(optimize_command COMMAND ${SPIRV_OPT_COMMAND} -Os "${out_file}" -o "${out_file}")
    endif()

    add_custom_command(OUTPUT ${out_file}
                       DEPENDS ${src_file}
                       DEPENDS ${SHADER_SOURCE_DEPENDENCIES}
                       DEPENDS ${COMPILE_ARGS_DEP}
                       MAIN_DEPENDENCY ${src_file}
                       COMMAND ${CMAKE_COMMAND} -E make_directory ${out_dir}
                       COMMAND ${GLSLANG_COMPILER} ${glslang_command_line}
                       ${optimize_command})

    set(${params_OUTPUT_FILE_LIST} ${${params_OUTPUT_FILE_LIST}} ${out_file} PARENT_SCOPE)
endfunction()
