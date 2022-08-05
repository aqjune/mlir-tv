func @sum(%mat: tensor<100x100xf64>) -> tensor<f64>
{
  %c0 = arith.constant 0.0: f64
  %i0 = arith.constant 0: index
  %i2 = arith.constant 2: index
  %zero = arith.constant -0.0 : f64
  %i = linalg.init_tensor [] : tensor<f64>
  %outty = linalg.fill(%zero, %i): f64, tensor<f64> -> tensor<f64>
  %mat_col = tensor.collapse_shape %mat [[0, 1]] : tensor<100x100xf64> into tensor<10000xf64>
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
