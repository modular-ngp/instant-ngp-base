cmake_minimum_required(VERSION 4.00)
project(instant-ngp-base VERSION 1.0.0 LANGUAGES CXX CUDA)

option(INSTANT_NGP_BUILD_TESTS "Build tests for instant-ngp-base" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA archs" FORCE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(instant-ngp-base)
target_sources(instant-ngp-base
        PRIVATE
        src/cuda/ngp.cuda.session.cu
        src/cuda/ngp.cuda.dataset.cu
        src/cuda/ngp.cuda.train.cu
)

include(FetchContent)
set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_NO_FWD_BWD OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)
set(TCNN_BUILD_USE_FAST_MATH ON CACHE BOOL "" FORCE)
set(TCNN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
FetchContent_Declare(
        tiny-cuda-nn
        GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
        GIT_TAG master
)
FetchContent_MakeAvailable(tiny-cuda-nn)
target_link_libraries(instant-ngp-base PRIVATE tiny-cuda-nn)

FetchContent_Declare(
        threadpool
        GIT_REPOSITORY https://github.com/modular-ngp/threadpool.git
        GIT_TAG master
)
FetchContent_MakeAvailable(threadpool)
target_link_libraries(instant-ngp-base PRIVATE threadpool::threadpool)

FetchContent_Declare(
        args
        GIT_REPOSITORY https://github.com/Taywee/args.git
        GIT_TAG master
)
FetchContent_MakeAvailable(args)
target_link_libraries(instant-ngp-base PRIVATE taywee::args)

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG master
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(instant-ngp-base PRIVATE nlohmann_json::nlohmann_json)

FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)
FetchContent_MakeAvailable(stb)
target_include_directories(instant-ngp-base PRIVATE ${stb_SOURCE_DIR})
set(stb_impl "${CMAKE_CURRENT_BINARY_DIR}/stb_image_impl.cpp")
file(WRITE ${stb_impl} "#define STB_IMAGE_IMPLEMENTATION\n#include <stb_image.h>\n")
target_sources(instant-ngp-base PRIVATE ${stb_impl})

if (INSTANT_NGP_BUILD_TESTS)
    enable_testing()
    add_executable(test_baseline test/test_baseline.cpp)
    target_link_libraries(test_baseline PRIVATE instant-ngp-base)
    add_test(NAME run_test_baseline COMMAND test_baseline)
endif ()