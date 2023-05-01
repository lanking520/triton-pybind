cmake_minimum_required (VERSION 3.18)
project(triton_pybind)

add_subdirectory(pybind11)

pybind11_add_module(triton_pybind SHARED src/triton_pybind.cpp)

target_compile_features(triton_pybind PRIVATE cxx_std_11)

target_compile_options(
        triton_pybind
        PRIVATE
        -Wall -Wextra -Wno-type-limits -Wno-unused-parameter -Wno-deprecated-declarations -Werror
)

set(CMAKE_SKIP_RPATH TRUE)
set_target_properties(triton_pybind PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")

target_include_directories(
        triton_pybind
        PRIVATE
        tritonserver/include
)

target_link_libraries(
        triton_pybind
        PRIVATE
        /opt/tritonserver/lib/libtritonserver.so
        pybind11::headers
)
