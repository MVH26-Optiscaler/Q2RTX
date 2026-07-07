
#
# ZLIB
#

if(USE_SYSTEM_ZLIB)
    find_package(ZLIB REQUIRED)

    add_library(z INTERFACE IMPORTED GLOBAL)
    target_link_libraries(z INTERFACE ZLIB::ZLIB)
else()
    add_subdirectory(zlib)
    target_include_directories(zlibstatic PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/zlib> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/zlib>)

    set_target_properties(zlib PROPERTIES FOLDER extern)
    set_target_properties(zlibstatic PROPERTIES FOLDER extern POSITION_INDEPENDENT_CODE ON)
    set_target_properties(minigzip PROPERTIES FOLDER extern)
    set_target_properties(example PROPERTIES FOLDER extern)

    add_library(z INTERFACE IMPORTED GLOBAL)
    target_link_libraries(z INTERFACE zlibstatic)
    target_include_directories(z INTERFACE "${ZLIB_INCLUDE_DIRS}")
endif()

#
# glslangValidator
#

if(CONFIG_BUILD_GLSLANG)
    option(BUILD_EXTERNAL OFF)
    option(ENABLE_SPVREMAPPER OFF)
    option(ENABLE_HLSL OFF)
    option(ENABLE_OPT OFF)
    option(ENABLE_PCH OFF)
    option(ENABLE_CTEST OFF)

    add_subdirectory(glslang)

    set_target_properties(glslang-standalone PROPERTIES FOLDER glslang)
endif()

#
# SDL 2
#

# On a 32-bit target, only the server is built, and it doesn't need SDL
if (IS_64_BIT)
  if (USE_SYSTEM_SDL2)
      find_package(SDL2 REQUIRED)

      add_library(SDL2 INTERFACE IMPORTED GLOBAL)
      target_link_libraries(SDL2 INTERFACE SDL2::SDL2)
    else()
      option(SDL_SHARED "" OFF)
      option(SDL_STATIC "" ON)
      option(SDL_SHARED_ENABLED_BY_DEFAULT OFF)
      option(SDL_TEST OFF)

      if (CONFIG_LINUX_STEAM_RUNTIME_SUPPORT)
          option(INPUT_TSLIB OFF)
      endif()

      add_subdirectory(SDL2)

      set_target_properties(SDL2main PROPERTIES FOLDER extern)
      set_target_properties(SDL2-static PROPERTIES FOLDER extern)
      set_target_properties(uninstall PROPERTIES FOLDER extern)
      set_target_properties(sdl_headers_copy PROPERTIES FOLDER extern)

      add_library(SDL2 INTERFACE IMPORTED GLOBAL)
      target_link_libraries(SDL2 INTERFACE SDL2main SDL2-static)
    endif()
endif()

#
# CURL
#
if(CONFIG_USE_CURL)
    if(USE_SYSTEM_CURL)
        find_package(CURL REQUIRED)
        add_library(libcurl INTERFACE IMPORTED GLOBAL)
        target_link_libraries(libcurl INTERFACE CURL::libcurl)
    else()
        option(BUILD_CURL_EXE "" OFF)
        option(BUILD_SHARED_LIBS "" OFF)
        option(CURL_STATICLIB "" ON)
        option(BUILD_TESTING "" OFF)
        option(ENABLE_MANUAL "" OFF)
        option(ENABLE_DOCS "" OFF)
        option(HTTP_ONLY "" ON)
        option(CURL_ZLIB "" OFF)
        option(CMAKE_USE_OPENSSL "" OFF)
        set(CURL_CA_PATH "none" CACHE PATH "")

        set(CMAKE_DEBUG_POSTFIX "" CACHE STRING "Override CMAKE_DEBUG_POSTFIX, which curl sets to '-d'")
    
        set(ZLIB_DIR "{CMAKE_CURRENT_SOURCE_DIR}/zlib")

        add_subdirectory(curl)

        set_target_properties(libcurl_static PROPERTIES FOLDER extern)
        if (TARGET libcurl_object)
            set_target_properties(libcurl_object PROPERTIES FOLDER extern)
        endif()
        set_target_properties(curl_uninstall PROPERTIES FOLDER extern)
    endif()
endif()


add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)

add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader-c)

#
# openal-soft (64-bit only, not needed for the 32-bit server)
#

if (IS_64_BIT)
    if (USE_SYSTEM_OPENAL)
        find_package(OpenAL REQUIRED)

        # Provide "OpenAL" target with result of CMake's detection
        add_library(OpenAL INTERFACE IMPORTED GLOBAL)
        target_link_libraries(OpenAL INTERFACE ${OPENAL_LIBRARY})
        target_include_directories(OpenAL INTERFACE ${OPENAL_INCLUDE_DIR})
    else()
        set(LIBTYPE STATIC)
        option(FORCE_STATIC_VCRT "" ON)
        option(ALSOFT_UTILS "" OFF)
        option(ALSOFT_EXAMPLES "" OFF)
        option(ALSOFT_UPDATE_BUILD_VERSION "" OFF)
        option(ALSOFT_INSTALL "" OFF)
        option(ALSOFT_INSTALL_CONFIG "" OFF)
        option(ALSOFT_INSTALL_HRTF_DATA "" OFF)
        option(ALSOFT_INSTALL_AMBDEC_PRESETS "" OFF)
        option(ALSOFT_INSTALL_EXAMPLES "" OFF)
        option(ALSOFT_INSTALL_UTILS "" OFF)
        option(ALSOFT_ENABLE_MODULES "" OFF)
        add_subdirectory(openal-soft)
        set_target_properties(OpenAL PROPERTIES FOLDER extern/openal-soft)
        set_target_properties(alsoft.common PROPERTIES FOLDER extern/openal-soft)
        set_target_properties(alsoft.excommon PROPERTIES FOLDER extern/openal-soft)
        set_target_properties(alsoft.fmt PROPERTIES FOLDER extern/openal-soft)
        set_target_properties(clang-tidy-check PROPERTIES FOLDER extern/openal-soft)
        target_compile_definitions(OpenAL PUBLIC AL_LIBTYPE_STATIC)
    endif()
endif()

#
# ONNX Runtime + Qualcomm QNN Execution Provider (NPU upscaler backend)
#
# Uses Microsoft's officially-supported classic/built-in QNN EP package,
# which bundles the Qualcomm QNN backend DLLs (QnnHtp.dll etc.) directly --
# no separate system-wide QAIRT SDK install is required to build or run.
# Windows-ARM64 only (Snapdragon Hexagon NPU); gated off everywhere else.
#

if(USE_ORT_QNN_UPSCALER)
    set(ONNXRUNTIME_QNN_VERSION "1.24.4")

    include(FetchContent)
    FetchContent_Declare(
        onnxruntime_qnn
        URL "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.QNN/${ONNXRUNTIME_QNN_VERSION}"
        DOWNLOAD_NAME "onnxruntime_qnn.zip"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime_qnn)

    set(ONNXRUNTIME_QNN_NATIVE_DIR "${onnxruntime_qnn_SOURCE_DIR}/runtimes/win-arm64/native")

    add_library(onnxruntime_qnn INTERFACE IMPORTED GLOBAL)
    target_include_directories(onnxruntime_qnn INTERFACE "${onnxruntime_qnn_SOURCE_DIR}/build/native/include")
    target_link_libraries(onnxruntime_qnn INTERFACE "${ONNXRUNTIME_QNN_NATIVE_DIR}/onnxruntime.lib")

    # Runtime payload that must be deployed next to the game executable:
    # onnxruntime.dll itself, its provider-bridge DLLs
    # (onnxruntime_providers_qnn.dll/_shared.dll), the QNN backend DLLs
    # (QnnHtp/QnnCpu/QnnGpu/QnnSystem.dll + HTP V73/V81 stub DLLs), and the
    # Hexagon-DSP-side skel binaries (libQnnHtpV7x/V8xSkel.so + .cat) that the
    # HTP backend pushes to the NPU at runtime. Glob the whole native
    # directory rather than hand-listing files, since missing any one of
    # these breaks QNN EP session creation. src/CMakeLists.txt copies this
    # set next to client.exe post-build.
    file(GLOB ONNXRUNTIME_QNN_RUNTIME_DLLS "${ONNXRUNTIME_QNN_NATIVE_DIR}/*.dll" "${ONNXRUNTIME_QNN_NATIVE_DIR}/*.so" "${ONNXRUNTIME_QNN_NATIVE_DIR}/*.cat")
    set(ONNXRUNTIME_QNN_RUNTIME_DLLS "${ONNXRUNTIME_QNN_RUNTIME_DLLS}" CACHE INTERNAL "ONNX Runtime QNN EP runtime payload to deploy next to client.exe")
endif()
