func @f(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<f32> {
  %outty = linalg.init_tensor [] : tensor<f32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
     outs(%outty : tensor<f32>) {
     ^bb0(%ai : f32, %bi: f32, %res : f32):
    %s = arith.mulf %ai, %bi: f32
    %res2 = arith.addf %s, %res : f32
    linalg.yield %res2 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
