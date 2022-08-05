func @gather_minimum_info(%arg0 : tensor<3x?x5xi32>, %arg1 : tensor<?x6xi32>) {
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x?x5xi32>, tensor<?x6xi32>)  -> (tensor<?x?x?xi32>)
  return
}
