// VERIFY
// ARGS: --use-neg-zero

// Without --use-neg-zero, this transformation is incorrect because tgt is filling
// a non-identity value (+0.0) to the output tensor.

func.func @f(%img: tensor<10x3xf32>, %weights: tensor<6x3xf32>, %bias: tensor<6xf32>) -> tensor<10x6xf32> {
  %0 = "tosa.fully_connected"(%img, %weights, %bias) : (tensor<10x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<10x6xf32>)
  return %0 : tensor<10x6xf32>
}
