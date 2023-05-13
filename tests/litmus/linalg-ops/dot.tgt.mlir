func.func @f(%a: tensor<100xf32>, %b: tensor<100xf32>) -> tensor<f32> {
  %outty = tensor.empty () : tensor<f32>
  %zero = arith.constant -0.0 : f32
  %filled = linalg.fill ins(%zero: f32) outs(%outty: tensor<f32>) -> tensor<f32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%a, %b : tensor<100xf32>, tensor<100xf32>)
     outs(%filled : tensor<f32>) {
     ^bb0(%ai : f32, %bi: f32, %res : f32):
    %s = arith.mulf %ai, %bi: f32
    %res2 = arith.addf %s, %res : f32
    linalg.yield %res2 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
