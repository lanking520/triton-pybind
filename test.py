from build import triton_pybind
import numpy as np

print(triton_pybind.add(1, 2))
result = triton_pybind.vector_test([1, 2, 3])
print(type(result), result)
result = triton_pybind.numpy_change_shape(np.ones(3, dtype="int32"))
print(type(result), result)
print(triton_pybind.play_string("Hello World!"))

print("Start Triton API changes")

triton_pybind.create_triton_server("/path/to/model", "/opt/tritonserver/backends",
                                   "/opt/tritonserver/repoagents", 0)
