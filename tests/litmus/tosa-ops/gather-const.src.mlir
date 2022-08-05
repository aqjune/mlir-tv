// VERIFY
func @test_gather() -> tensor<2x1x1xf32> {
  %v = linalg.init_tensor [2,2,1]: tensor<2x2x1xf32>
  %zero = arith.constant 0: index
  %one = arith.constant 1: index
  %c1 = arith.constant 10.0: f32
  %c2 = arith.constant 11.0: f32
  %v1 = tensor.insert %c1 into %v [%zero, %zero, %zero]: tensor<2x2x1xf32>
  %v2 = tensor.insert %c2 into %v1[%one,  %zero, %zero]: tensor<2x2x1xf32>

  %indices = arith.constant dense <[[0], [0]]>: tensor<2x1xi32>

  %0 = "tosa.gather"(%v2, %indices) : (tensor<2x2x1xf32>, tensor<2x1xi32>) -> tensor<2x1x1xf32>
  return %0 : tensor<2x1x1xf32>
}
