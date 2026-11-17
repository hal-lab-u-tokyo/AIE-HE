# AIE-HE's host with its Encode connected to its Encrypt (aiehe_joined.h): the
# host source AIEHE_SRC (planB's or planA's test.cpp) built with ../ckks_encode's
# encoder and SEAL.
cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../common.cmake)

set(CMAKE_C_COMPILER gcc-13)
set(CMAKE_CXX_COMPILER g++-13)
set(XRT_INC_DIR /opt/xilinx/xrt/include CACHE STRING "Path to XRT include")
set(XRT_LIB_DIR /opt/xilinx/xrt/lib CACHE STRING "Path to xrt_coreutil")
set(TARGET_NAME kernel CACHE STRING "Target to be built")
set(AIEHE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../ckks_encrypt_planB/test.cpp CACHE FILEPATH "AIE-HE host source")

project(${TARGET_NAME})

# The same flag as the SEAL build.
add_compile_options(-march=native)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ckks_encrypt_SEAL/thirdparty/SEAL SEAL)

get_filename_component(aiehe_dir ${AIEHE_SRC} DIRECTORY)
add_executable(${TARGET_NAME} ${AIEHE_SRC} ${aiehe_dir}/../nttlib.cpp ../ckks_encode/ckks_encoder.cpp blake2/blake2b.c blake2/blake2xb.c)
target_include_directories(${TARGET_NAME} PUBLIC ${XRT_INC_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ../ckks_encode)
target_link_directories(${TARGET_NAME} PUBLIC ${XRT_LIB_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC xrt_coreutil SEAL::seal)
target_link_test_utils(${TARGET_NAME})
