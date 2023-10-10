import monotonic_alignment_cpp
import numpy as np
import time

np.set_printoptions(precision=3, suppress=True)
np.random.seed(42)

if monotonic_alignment_cpp.check_openmp():
    print("OpenMP is available")
else:
    print("OpenMP is not available")

batch_size = 64

t_x = np.random.randint(200,400, size=(batch_size,))
t_y_multipliers = np.random.uniform(1.5, 3.0, size=(batch_size,))
t_y = (t_x * t_y_multipliers).astype(np.int64)

max_t_x = np.max(t_x)
max_t_y = np.max(t_y)

batch = np.random.randn(batch_size, max_t_x, max_t_y)

batch -= np.max(batch, axis=(1,2)).reshape(-1,1,1) + 1.0

x_mask = np.arange(max_t_x).reshape(1,-1) < t_x.reshape(-1,1)
y_mask = np.arange(max_t_y).reshape(1,-1) < t_y.reshape(-1,1)

mask = np.matmul(x_mask.reshape(batch_size, max_t_x, 1), y_mask.reshape(batch_size, 1, max_t_y))

def maximum_path_cpp_wrapper(batch, mask):
    t_x = mask[:, :, 0].sum(axis=1)
    t_y = mask[:, 0, :].sum(axis=1)
    path = np.zeros_like(batch, dtype=np.float32)

    monotonic_alignment_cpp.maximum_path(batch, t_x, t_y, path)

    return path

def maximum_path_numpy(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    dtype = value.dtype
    mask = mask.astype(bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = path.astype(dtype)
    return path

start_time = time.time()
path_cpp = maximum_path_cpp_wrapper(batch, mask)
cpp_time = time.time() - start_time

start_time = time.time()
path_numpy = maximum_path_numpy(batch, mask)
numpy_time = time.time() - start_time

print(np.allclose(path_cpp, path_numpy))

print(f"CPP time: {cpp_time}")
print(f"NumPy time: {numpy_time}")