#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cassert>
#include <iostream>

namespace py = pybind11;

pybind11::array_t<float> increment(pybind11::array_t<float, py::array::c_style | py::array::forcecast> input) {
    auto buf = input.request();
    float *ptr = (float *) buf.ptr;

    assert(buf.ndim == 3);

    int batch_size = buf.shape[0];
    int t_x = buf.shape[1];
    int t_y = buf.shape[2];
    float (*arr)[t_x][t_y] = (float (*)[t_x][t_y]) ptr;

    // Create output array
    py::array_t<float> output = py::array_t<float>(buf.size);
    // Set shape of output array
    output.resize({batch_size, t_x, t_y});
    auto buf_out = output.request();
    float *ptr_out = (float *) buf_out.ptr;
    float (*arr_out)[t_x][t_y] = (float (*)[t_x][t_y]) ptr_out;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < t_x; j++) {
            for (int k = 0; k < t_y; k++) {
                arr_out[i][j][k] = arr[i][j][k] + 1.0f;
            }
        }
    }

    return output;
}

PYBIND11_MODULE(monotonic_alignment_cpp, m) {
    m.doc() = "Monotonic alignment written in C++"; // optional module docstring

    m.def("increment", &increment, "A function that increments all elements of a numpy array by 1", py::arg("input"));
}