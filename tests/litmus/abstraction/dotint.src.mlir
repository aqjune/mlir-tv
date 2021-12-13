// EXPECT: "AbsLevelIntDot: SUM_MUL"

func @f(%a: tensor<?xi32>, %b: tensor<?xi32>) -> tensor<i32> {
  %i = linalg.init_tensor []: tensor<i32>
  %e = linalg.dot ins(%a, %b : tensor<?xi32>,tensor<?xi32>)
    outs(%i: tensor<i32>) -> tensor<i32>
  return %e : tensor<i32>
}
