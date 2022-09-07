func.func private @linear_tensor(%t: tensor<5xf32>) -> f32

func.func @different_tensor(%v: tensor<5xf32>) -> f32 {
  %i = arith.constant 4: index
  %c = arith.constant 3.0: f32
  %mv = tensor.insert %c into %v[%i]: tensor<5xf32>
  %r = func.call @linear_tensor(%mv): (tensor<5xf32>) -> f32
  return %r: f32
}
