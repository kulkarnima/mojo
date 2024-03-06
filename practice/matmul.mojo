from random import rand

import benchmark
from memory import memset_zero
from python import Python

alias M = 512
alias N = 4096
alias K = 512
alias type = DType.float32

# simwidth of = amount of `type` elements that fit into a single SIMD register
# 2x multiplier will use multiple SIMD registers in parallel where possible
alias nelts = simwidthof[type]() * 2
alias tile_n = 64 # N must be multiple of this
alias tile_k = 4 # K must be multiple of this


struct Matrix[rows: Int, cols: Int]:
  var data: DTypePointer[type]

  fn __init__(inout self):
    self.data = DTypePointer[type].alloc(rows * cols)
    memset_zero(self, data, rows * cols)

  fn __init__(inout self, data: DTypePointer[type]):
    self.data = data

  @staticmethod
  fn rand() -> Self:
    var data = DTypePointer[type].alloc(rows * cols)
    rand(data, rows * cols)
    return Self(data)

  fn __getitem__(self, y: Int, x: Int) -> SIMD[type, 1]:
    return self.load[1](y, x)

  fn __setitem__(self, y: Int, x: Int, val: SIMD[type, 1]):
    self.store[1](y, x, val)

  fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
    return self.data.simd_load[nelts](y * self.cols + x)

  fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
    return self.data.simd_store[nelts](y * self.cols + x, val)


def run_matmul_python() -> Float64:
  var pymatmul: PythonObject = Python.import_module("pymatmul")
  var py = Python.import_module("builtins")

  var gflops = pymatmul.benchmark_matmul_numpy(M, N, K).to_float64()
  py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Numpy:", gflops))

  return gflops

fn matmul_naive(inout C: Matrix, A: Matrix, B: Matrix):
  for m in range(C.rows):
    for k in range(A.cols):
      for n in range(C.cols):
        C[m, n] += A[m, k] * B[k, n]


fn test_all() raises:
  var A = Matrix[M, K].rand()
  var B = Matrix[K, N].rand()
  var C = Matrix[M, N]()

  matmul_naive(C, A, B)

  A.data.free()
  B.data.free()
  C.data.free()

fn main() raises:
  constrained[N % tile_n == 0, "N must of multiple of tile_n"]()
  constrained[K % tile_k == 0, "K must be mutliple of tile_k"]()

  test_all()
  print("CPU Results\n")
  var python_gflops = run_matmul_python()
  var numpy_gflops = run_matmul_numpy()

  bench[matmul_naive, "Naive:"](python_gflops, numpy_gflops)
