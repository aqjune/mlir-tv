// VERIFY

func.func @f(%v: tensor<2xi32>, %w: tensor<2xi32>) -> tensor<2xi32> {
  %x = arith.addi %v, %w: tensor<2xi32>
  return %x: tensor<2xi32>
}
