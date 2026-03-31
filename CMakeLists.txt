cmake_minimum_required(VERSION 3.22)
project(niftyreg_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(BUILD_ALL_DEP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(USE_OPENCL OFF CACHE BOOL "" FORCE)
set(USE_OPENMP ON CACHE BOOL "" FORCE)
set(USE_NRRD OFF CACHE BOOL "" FORCE)

option(NIFTYREG_PYTHON_USE_CUDA "Build NiftyReg with CUDA support" OFF)
option(NIFTYREG_PYTHON_CHECK_GPU "Enable NiftyReg GPU availability checks" OFF)

set(USE_CUDA ${NIFTYREG_PYTHON_USE_CUDA} CACHE BOOL "" FORCE)
set(CHECK_GPU ${NIFTYREG_PYTHON_CHECK_GPU} CACHE BOOL "" FORCE)

set(NIFTYREG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/niftyreg)

add_subdirectory(${NIFTYREG_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/niftyreg-build)

pybind11_add_module(_niftyreg src/niftyreg_pybind.cpp)

if(USE_CUDA)
  target_compile_definitions(_niftyreg PRIVATE USE_CUDA)
endif()
if(USE_OPENCL)
  target_compile_definitions(_niftyreg PRIVATE USE_OPENCL)
endif()

target_include_directories(_niftyreg
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/niftyreg-build
    ${NIFTYREG_SOURCE_DIR}/reg-io
    ${NIFTYREG_SOURCE_DIR}/reg-io/png
    ${NIFTYREG_SOURCE_DIR}/third-party
    ${NIFTYREG_SOURCE_DIR}/reg-lib
    ${NIFTYREG_SOURCE_DIR}/reg-lib/cpu
)

target_link_libraries(_niftyreg
  PRIVATE
    _reg_aladin
    _reg_f3d
    _reg_ReadWriteImage
    _reg_tools
)

install(TARGETS _niftyreg LIBRARY DESTINATION niftyreg)
