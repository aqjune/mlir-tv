func.func @test_gather() -> tensor<2x1x1xf32> {
  %res = arith.constant dense <[[[10.0]], [[12.0]]]>: tensor<2x1x1xf32>
  return %res: tensor<2x1x1xf32>
}
