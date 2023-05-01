#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tritonserver.h"
#include <iostream>

namespace py = pybind11;

#define PrintErrorMsg(cond, error_msg) \
    if (!cond) { \
      std::cerr << "error: " << (error_msg); \
    }

int add(int i, int j) {
    return i + j;
}

py::array_t<int> vector_test(std::vector<int> &v) {
    return py::array_t<int>(
            {1, 3}, // shape
            v.data());
}

py::array_t<size_t> numpy_change_shape(py::array_t<size_t> &v) {
    return py::array_t<size_t>(
            {1, 1, 3}, // shape
            v.data());
}

std::string play_string(const char* inputs) {
    std::string converted(inputs);
    return converted;
}


uintptr_t create_triton_server(char* repo_path, char* backends_path, char* agent_path, int verbose) {
    TRITONSERVER_ServerOptions* server_options_ptr = nullptr;
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsNew(&server_options_ptr), "creating server options");
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                    server_options_ptr, repo_path),
            "setting model repository path");
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsSetLogVerbose(server_options_ptr, verbose),
            "setting verbose logging level");
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsSetBackendDirectory(
                    server_options_ptr, backends_path),
            "setting backend directory");
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
                    server_options_ptr, agent_path),
            "setting repository agent directory");
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options_ptr, true),
            "setting strict model configuration");
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsSetModelControlMode(
                    server_options_ptr, TRITONSERVER_MODEL_CONTROL_EXPLICIT),
            "setting model control");
    double min_compute_capability = 0;
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
                    server_options_ptr, min_compute_capability),
            "setting minimum supported CUDA compute capability");
    TRITONSERVER_Server* server_ptr = nullptr;
    PrintErrorMsg(
            TRITONSERVER_ServerNew(&server_ptr, server_options_ptr),
            "creating server object")
    PrintErrorMsg(
            TRITONSERVER_ServerOptionsDelete(server_options_ptr), "deleting server options")
    return reinterpret_cast<uintptr_t>(server_ptr);
}


PYBIND11_MODULE(triton_pybind, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("vector_test", &vector_test, R"pbdoc(
        Test vector functions.
    )pbdoc");

    m.def("numpy_change_shape", &numpy_change_shape, R"pbdoc(
        Test change numpy array shape.
    )pbdoc");

    m.def("play_string", &play_string, R"pbdoc(
        Play String element
    )pbdoc");

    m.def("create_triton_server", &create_triton_server, R"pbdoc(
        Create Triton Server
    )pbdoc");
}
