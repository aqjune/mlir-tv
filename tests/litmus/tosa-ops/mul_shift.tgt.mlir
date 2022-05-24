func.func @f(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = "tosa.mul"(%arg1, %arg0) { shift = 1: i32 } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}

