// VERIFY

func @f(%arg0: index) -> index {
  %result = tensor.generate %arg0 {
  ^bb0(%i: index, %j: index):
    %sum = arith.addi %i, %j : index
    tensor.yield %sum : index
  } : tensor<16x?xindex>
  %c1 = arith.constant 1: index
  %d = tensor.dim %result, %c1: tensor<16x?xindex>
  return %d: index
}
