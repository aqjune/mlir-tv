func.func @f(%t: tensor<?x?xf32>, %pad_value: f32) -> tensor<?x?xf32>{
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c3 = arith.constant 3: index
  %c5 = arith.constant 5: index
  %d0 = tensor.dim %t, %c0: tensor<?x?xf32>
  %d1 = tensor.dim %t, %c1: tensor<?x?xf32>

  %newd0 = arith.addi %d0, %c3: index
  %newd1 = arith.addi %d1, %c5: index

  %result = tensor.generate %newd0, %newd1 {
  ^bb0(%i: index, %j: index):
    tensor.yield %pad_value : f32
  } : tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
