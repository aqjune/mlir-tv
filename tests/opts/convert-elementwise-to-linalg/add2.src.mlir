// VERIFY

func.func @f(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<?xf32>
  return %0 : tensor<?xf32>
}
