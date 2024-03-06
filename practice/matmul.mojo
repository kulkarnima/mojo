from random import rand

import benchmark
from algorithm import parallelize, vectorize
from memory import memset_zero
from python import Python

alias M = 512  # rows of A and C
alias N = 4096  # cols of B and C
alias K = 512  # cols of A and rows of B
alias type = DType.float32

# simdwidth of = amount of `type` elements that fit into a single SIMD register
# 2x multiplier will use multiple SIMD registers in parallel where possible
alias nelts = simdwidthof[type]() * 2
alias tile_n = 64  # N must be a multiple of this
alias tile_k = 4  # K must be a multiple of this


struct Matrix[rows: Int, cols: Int]:
  var data: DTypePointer[type]

  # Initialize zeroeing all values
  fn __init__(inout self):
    self.data = DTypePointer[type].alloc(rows * cols)
    memset_zero(self.data, rows * cols)

  # Initialize taking a pointer, don't set any elements
  fn __init__(inout self, data: DTypePointer[type]):
    self.data = data

  ## Initialize with random values
  @staticmethod
  fn rand() -> Self:
    var data = DTypePointer[type].alloc(rows * cols)
    rand(data, rows * cols)
    return Self(data)

  fn __getitem__(self, y: Int, x: Int) -> SIMD[type, 1]:
    return self.load[1](y, x)

  fn __setitem__(inout self, y: Int, x: Int, val: SIMD[type, 1]):
    self.store[1](y, x, val)

  fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
    return self.data.simd_load[nelts](y * self.cols + x)

  fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
    return self.data.simd_store[nelts](y * self.cols + x, val)


def run_matmul_python() -> Float64:
  Python.add_to_path(".")
  var pymatmul: PythonObject = Python.import_module("pymatmul")
  var py = Python.import_module("builtins")

  var gflops = pymatmul.benchmark_matmul_python(128, 128, 128).to_float64()
  py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Python:", gflops))

  return gflops

def run_matmul_numpy() -> Float64:
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


# Using stdlib vectorize function
fn matmul_vectorized(inout C: Matrix, A: Matrix, B: Matrix):
  for m in range(C.rows):
    for k in range(A.cols):

      @parameter
      fn dot[nelts: Int](n: Int):
        C.store[nelts](
            m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
        )

      vectorize[dot, nelts, size = C.cols]()

@always_inline
fn bench[
    func: fn(inout Matrix, Matrix, Matrix) -> None, name: StringLiteral
](base_gflops: Float64, numpy_gflops: Float64) raises:
  var A = Matrix[M, K].rand()
  var B = Matrix[K, N].rand()
  var C = Matrix[M, N]()

  @always_inline
  @parameter
  fn test_fn():
    _ = func(C, A, B)

  var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()

  A.data.free()
  B.data.free()
  C.data.free()

  var gflops = ((2 * M * N * K) / secs) / 1e9
  var speedup: Float64 = gflops / base_gflops
  var numpy_speedup: Float64 = gflops / numpy_gflops

  var py = Python.import_module("builtins")
  _ = py.print(
      py.str("{:<13}{:>8.3f} GFLOPS {:>9.2f}x Python {:>5.2f}x Numpy").format(
        name, gflops, speedup, numpy_speedup
      )
  )

@always_inline
fn test_matrix_equal[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](inout C: Matrix, A: Matrix, B: Matrix) raises -> Bool:
  """Runs a matmul function on A and B and tests the result for euality with
  C on every element.
  """
  var result = Matrix[M, N]()
  _ = func(result, A, B)
  for i in range(C.rows):
    for j in range(C.cols):
      if C[i, j] != result[i, j]:
        return False
  return True

fn test_all() raises:
  var A = Matrix[M, K].rand()
  var B = Matrix[K, N].rand()
  var C = Matrix[M, N]()

  matmul_naive(C, A, B)

  A.data.free()
  B.data.free()
  C.data.free()



fn main() raises:
  constrained[N % tile_n == 0, "N must be a multiple of tile_n"]()
  constrained[K % tile_k == 0, "K must be a multiple of tile_k"]()

  test_all()
  print("CPU Results\n")
  var python_gflops = run_matmul_python()
  var numpy_gflops = run_matmul_numpy()

  bench[matmul_naive, "Naive: "](python_gflops, numpy_gflops)
  bench[matmul_vectorized, "Vectorized: "](python_gflops, numpy_gflops)
