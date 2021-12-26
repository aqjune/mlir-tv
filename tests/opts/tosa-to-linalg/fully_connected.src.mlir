// VERIFY-INCORRECT

// Interestingly, this transformation FAILS to verify because filling
// non-identity value (+0.0) to output tensor.

func @f(%img: tensor<5x3xf32>, %weights: tensor<6x3xf32>, %bias: tensor<6xf32>) -> tensor<5x6xf32> {
  %0 = "tosa.fully_connected"(%img, %weights, %bias) : (tensor<5x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<5x6xf32>)
  return %0 : tensor<5x6xf32>
}
