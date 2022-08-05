func @f(%arg0: index) -> tensor<16x?xindex> {
  %result = tensor.generate %arg0 {
  ^bb0(%i: index, %j: index):
    %sum = arith.subi %i, %j : index
    tensor.yield %sum : index
  } : tensor<16x?xindex>
  return %result : tensor<16x?xindex>
}
