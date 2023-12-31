#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cassert>
#include <thread>

namespace py = pybind11;

void maximum_path_one(float *values_ptr, float *path_ptr, int t_x, int t_y, int max_t_y) {
    // Convert ptrs to 2D array
    float (*values)[max_t_y] = (float (*)[max_t_y]) values_ptr;
    float (*path)[max_t_y] = (float (*)[max_t_y]) path_ptr;

    for(int y = 1; y < t_y; y++) {
        values[0][y] += values[0][y - 1];
    }

    // Cannot start from x other than 0 (because path starts from 0, 0), so fill with negative infinity
    for(int x = 1; x < t_x; x++) {
        values[x][0] = -std::numeric_limits<float>::infinity();
    }

    for(int x = 1; x < t_x; x++) {
        for(int y = 1; y < t_y; y++) {
            values[x][y] += std::max(values[x - 1][y - 1], values[x][y - 1]);
        }
    }

    // Restore path
    int x = t_x - 1;
    for(int y = t_y - 1; y >= 0; y--) {
        path[x][y] = 1.0f;
        if(x > 0 && y > 0) {
            if(values[x - 1][y - 1] > values[x][y - 1]) {
                x--;
            }
        }
    }
}

void maximum_path(
    pybind11::array_t<float, py::array::c_style | py::array::forcecast> values,
    pybind11::array_t<int, py::array::c_style | py::array::forcecast> t_x,
    pybind11::array_t<int, py::array::c_style | py::array::forcecast> t_y,
    pybind11::array_t<float, py::array::c_style> path
) {
    auto buf_values = values.request();
    float *ptr_values = (float *) buf_values.ptr;

    auto buf_t_x = t_x.request();
    int *ptr_t_x = (int *) buf_t_x.ptr;

    auto buf_t_y = t_y.request();
    int *ptr_t_y = (int *) buf_t_y.ptr;

    auto buf_path = path.request();
    float *ptr_path = (float *) buf_path.ptr;

    assert(buf_values.ndim == 3);
    assert(buf_path.ndim == 3);
    assert(buf_t_x.ndim == 1);
    assert(buf_t_y.ndim == 1);

    assert(buf_values.shape[0] == buf_path.shape[0]);
    assert(buf_values.shape[1] == buf_path.shape[1]);
    assert(buf_values.shape[2] == buf_path.shape[2]);

    int batch_size = buf_values.shape[0];
    int max_t_x = buf_values.shape[1];
    int max_t_y = buf_values.shape[2];

    assert(buf_values.shape[0] == buf_t_x.shape[0]);
    assert(buf_values.shape[0] == buf_t_y.shape[0]);

    for(int i = 0; i < batch_size; i++) {
        assert(ptr_t_x[i] <= max_t_x);
        assert(ptr_t_y[i] <= max_t_y);
    }

    #pragma omp parallel for
    for(int bid = 0; bid < batch_size; bid++) {
        // Compute maximum path
        maximum_path_one(
            ptr_values + bid * max_t_x * max_t_y,
            ptr_path + bid * max_t_x * max_t_y,
            ptr_t_x[bid],
            ptr_t_y[bid],
            max_t_y
        );
    }
}

bool check_openmp() {
    # ifdef _OPENMP
        return true;
    # else
        return false;
    # endif
}

PYBIND11_MODULE(monotonic_alignment_cpp, m) {
    m.doc() = "Monotonic alignment written in C++"; // optional module docstring

    m.def(
        "maximum_path",
        &maximum_path,
        "A function that finds a maximum path in a batch of values",
        py::arg("values"),
        py::arg("t_x"),
        py::arg("t_y"),
        py::arg("path"),
        R"pbdoc(
            A function that finds a maximum path in a batch of values

            Args:
                values (np.ndarray): A batch of values
                t_x (np.ndarray): A batch of x coordinates of the last point in the path
                t_y (np.ndarray): A batch of y coordinates of the last point in the path
                path (np.ndarray): A batch of paths to store output in
        )pbdoc"
    );

    m.def(
        "check_openmp",
        &check_openmp,
        "A function that checks if OpenMP is enabled"
    );
}