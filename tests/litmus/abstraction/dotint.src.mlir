// EXPECT: "dot ops (int): SUM_MUL"

func.func @f(%a: tensor<?xi32>, %b: tensor<?xi32>) -> tensor<i32> {
  %zero = arith.constant 0 : i32
  %i = tensor.empty (): tensor<i32>
  %outty = linalg.fill ins(%zero: i32) outs(%i: tensor<i32>) -> tensor<i32>
  %e = linalg.dot ins(%a, %b : tensor<?xi32>,tensor<?xi32>)
    outs(%outty: tensor<i32>) -> tensor<i32>
  return %e : tensor<i32>
}
