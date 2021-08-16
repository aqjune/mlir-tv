// VERIFY

func @f() -> f32 {
  %one = constant 1: index
  %two = constant 2: index
  %c = constant sparse<[[1, 2], [3, 4]], [1.0, 2.0]> : tensor<4x8xf32>
  %onefloat = tensor.extract %c[%one, %two] : tensor<4x8xf32>
  return %onefloat: f32
}
