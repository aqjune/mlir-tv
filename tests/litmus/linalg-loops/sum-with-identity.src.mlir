// EXPECT: "[fpSum]: Sum of array unrolled to fp_add."
// ARGS: --verbose

func @sum() -> tensor<f32>
{
  %outty = linalg.init_tensor [] : tensor<f32>
  %cst = arith.constant sparse<[[0], [1], [2], [3], [4]], [-1.200000e+01, -0.000000e+00, 3.000000e+00, 2.000000e+00, -0.000000e+00]> : tensor<5xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%cst : tensor<5xf32>) outs(%outty : tensor<f32>) {
     ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
