// EXPECT: "dot ops (int): SUM_MUL"

func @f(%a: tensor<?xi32>, %b: tensor<?xi32>) -> tensor<i32> {
  %zero = arith.constant 0 : i32
  %i = linalg.init_tensor []: tensor<i32>
  %outty = linalg.fill(%zero, %i) : i32, tensor<i32> -> tensor<i32>
  %e = linalg.dot ins(%a, %b : tensor<?xi32>,tensor<?xi32>)
    outs(%outty: tensor<i32>) -> tensor<i32>
  return %e : tensor<i32>
}
