#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cassert>

namespace py = pybind11;

pybind11::array_t<float> maximum_path(
    pybind11::array_t<float, py::array::c_style | py::array::forcecast> values,
    pybind11::array_t<int, py::array::c_style | py::array::forcecast> t_x,
    pybind11::array_t<int, py::array::c_style | py::array::forcecast> t_y
) {
    auto buf_values = values.request();
    float *ptr_values = (float *) buf_values.ptr;

    auto buf_t_x = t_x.request();
    int *ptr_t_x = (int *) buf_t_x.ptr;

    auto buf_t_y = t_y.request();
    int *ptr_t_y = (int *) buf_t_y.ptr;

    assert(buf_values.ndim == 3);
    assert(buf_t_x.ndim == 1);
    assert(buf_t_y.ndim == 1);

    int batch_size = buf_values.shape[0];
    int max_t_x = buf_values.shape[1];
    int max_t_y = buf_values.shape[2];

    assert(buf_values.shape[0] == buf_t_x.shape[0]);
    assert(buf_values.shape[0] == buf_t_y.shape[0]);

    for(int i = 0; i < batch_size; i++) {
        assert(ptr_t_x[i] <= max_t_x);
        assert(ptr_t_y[i] <= max_t_y);
    }

    // Clone the values into new float array
    float *values_clone = new float[buf_values.size];
    std::copy(ptr_values, ptr_values + buf_values.size, values_clone);

    // Convert values to 3D array
    float (*arr_values)[max_t_x][max_t_y] = (float (*)[max_t_x][max_t_y]) values_clone;

    // Create output array
    py::array_t<float> path = py::array_t<float>(buf_values.size);
    // Set shape of output array
    path.resize({batch_size, max_t_x, max_t_y});
    auto buf_path = path.request();
    float *ptr_path = (float *) buf_path.ptr;

    // Fill with zeros
    std::fill(ptr_path, ptr_path + buf_path.size, 0.0f);

    // Convert path to 3D array
    float (*arr_path)[max_t_x][max_t_y] = (float (*)[max_t_x][max_t_y]) ptr_path;

    for(int bid = 0; bid < batch_size; bid++) {
        // Compute maximum path

        for(int y = 1; y < ptr_t_y[bid]; y++) {
            arr_values[bid][0][y] += arr_values[bid][0][y - 1];
        }

        // Cannot start from x other than 0 (because path starts from 0, 0), so fill with negative infinity
        for(int x = 1; x < ptr_t_x[bid]; x++) {
            arr_values[bid][x][0] = -std::numeric_limits<float>::infinity();
        }

        for(int x = 1; x < ptr_t_x[bid]; x++) {
            for(int y = 1; y < ptr_t_y[bid]; y++) {
                arr_values[bid][x][y] += std::max(arr_values[bid][x - 1][y - 1], arr_values[bid][x][y - 1]);
            }
        }

        // Restore path
        int x = ptr_t_x[bid] - 1;
        for(int y = ptr_t_y[bid] - 1; y >= 0; y--) {
            arr_path[bid][x][y] = 1.0f;
            if(x > 0 && y > 0) {
                if(arr_values[bid][x - 1][y - 1] > arr_values[bid][x][y - 1]) {
                    x--;
                }
            }
        }
    }
    
    delete[] values_clone;

    return path;
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
        py::return_value_policy::move,
        R"pbdoc(
            A function that finds a maximum path in a batch of values

            Args:
                values (np.ndarray): A batch of values
                t_x (np.ndarray): A batch of x coordinates of the last point in the path
                t_y (np.ndarray): A batch of y coordinates of the last point in the path

            Returns:
                np.ndarray: A batch of paths
        )pbdoc"
    );
}