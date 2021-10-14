func @f(%a: tensor<?xi32>, %b: tensor<?xi32>) -> tensor<i32> {
  %outty = linalg.init_tensor [] : tensor<i32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%a, %b : tensor<?xi32>, tensor<?xi32>)
     outs(%outty : tensor<i32>) {
     ^bb0(%ai : i32, %bi: i32, %res : i32):
    %s = muli %ai, %bi: i32
    %res2 = addi %s, %res : i32
    linalg.yield %res2 : i32
  } -> tensor<i32>
  return %result : tensor<i32>
}
