// VERIFY
// ARGS: --associative

func @sum(%mat: tensor<100x100xf32>) -> tensor<f32>
{
  %zero = arith.constant -0.0 : f32
  %i = linalg.init_tensor [] : tensor<f32>
  %outty = linalg.fill(%zero, %i): f32, tensor<f32> -> tensor<f32>

  %mat_transposed = linalg.generic {
      indexing_maps = [affine_map<(d0,d1) -> (d1,d0)>,
                       affine_map<(d0,d1) -> (d0,d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mat: tensor<100x100xf32>)
      outs(%mat : tensor<100x100xf32>) {
    ^bb0(%item: f32, %out_dummy: f32):
      linalg.yield %item : f32
  } -> tensor<100x100xf32>


  %mat_col = linalg.tensor_collapse_shape %mat_transposed [[0, 1]] : tensor<100x100xf32> into tensor<10000xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
     ins(%mat_col : tensor<10000xf32>) outs(%outty : tensor<f32>) {
     ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
