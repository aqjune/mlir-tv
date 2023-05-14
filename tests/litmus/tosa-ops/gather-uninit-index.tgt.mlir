func.func @test_gather() -> tensor<2x1x1xf32> {
  %v = tensor.empty (): tensor<2x1x1xf32>
  return %v: tensor<2x1x1xf32>
}
