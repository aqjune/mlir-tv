func.func @f(%arg0: index) -> tensor<16x?xindex> {
  %result = tensor.generate %arg0 {
  ^bb0(%i: index, %j: index):
    %c2 = arith.constant 2: index
    %tw = arith.muli %j, %c2 : index
    %tmp = arith.subi %i, %j : index
    %sum = arith.addi %tmp, %tw : index
    tensor.yield %sum : index
  } : tensor<16x?xindex>
  return %result : tensor<16x?xindex>
}
