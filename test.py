import monotonic_alignment_cpp
import numpy as np

batch_size = 2
t_x = 4
t_y = 10

batch = np.arange(batch_size * t_x * t_y).reshape(batch_size, t_x, t_y).astype(np.float32)

print(batch)

print(monotonic_alignment_cpp.increment(batch))

print(batch)