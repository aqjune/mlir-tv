// VERIFY

func.func @f() -> f32 {
  %one = arith.constant 1: index
  %two = arith.constant 2: index
  %c = arith.constant sparse<[[1, 2], [3, 4]], [1.0, 2.0]> : tensor<4x8xf32>
  %onefloat = tensor.extract %c[%one, %two] : tensor<4x8xf32>
  return %onefloat: f32
}
