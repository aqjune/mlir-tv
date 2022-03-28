func @sum() -> tensor<f32>
{
  %zero = arith.constant -0.0 : f32
  %i = linalg.init_tensor [] : tensor<f32>
  %outty = linalg.fill ins(%zero: f32) outs(%i: tensor<f32>) -> tensor<f32>
  %cst = arith.constant sparse<[[0], [1], [2]], [-1.200000e+01, 3.000000e+00, 2.000000e+00]> : tensor<3xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%cst : tensor<3xf32>) outs(%outty : tensor<f32>) {
     ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
