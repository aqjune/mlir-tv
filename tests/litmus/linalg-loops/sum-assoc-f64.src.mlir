// VERIFY
// ARGS: --associative

func @sum(%mat: tensor<100x100xf64>) -> tensor<f64>
{
  %zero = arith.constant -0.0 : f64
  %i = linalg.init_tensor [] : tensor<f64>
  %outty = linalg.fill ins(%zero: f64) outs(%i: tensor<f64>) -> tensor<f64>

  %mat_transposed = linalg.generic {
      indexing_maps = [affine_map<(d0,d1) -> (d1,d0)>,
                       affine_map<(d0,d1) -> (d0,d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mat: tensor<100x100xf64>)
      outs(%mat : tensor<100x100xf64>) {
    ^bb0(%item: f64, %out_dummy: f64):
      linalg.yield %item : f64
  } -> tensor<100x100xf64>


  %mat_col = tensor.collapse_shape %mat_transposed [[0, 1]] : tensor<100x100xf64> into tensor<10000xf64>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%mat_col : tensor<10000xf64>) outs(%outty : tensor<f64>) {
     ^bb0(%arg0 : f64, %arg1 : f64):
        %0 = arith.addf %arg0, %arg1 : f64
        linalg.yield %0 : f64
  } -> tensor<f64>
  return %result : tensor<f64>
}
