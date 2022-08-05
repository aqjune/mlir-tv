// VERIFY

func @gather_float(%arg0: tensor<2x3x2xf32>, %arg1: tensor<2x3xi32>) -> () {
  %0 = "tosa.gather"(%arg0, %arg1)  : (tensor<2x3x2xf32>, tensor<2x3xi32>)  -> (tensor<2x3x2xf32>)
  return
}

// mlir-opt --tosa-to-linalg-on-tensors
