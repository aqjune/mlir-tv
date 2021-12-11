func @sum(%mat0: tensor<100x100xf32>) -> tensor<f32>
{
  %c0 = arith.constant 0.0: f32
  %i0 = arith.constant 0: index
  %i2 = arith.constant 2: index
  %mat = tensor.insert %c0 into %mat0[%i0,%i2]: tensor<100x100xf32>
  %outty = linalg.init_tensor [] : tensor<f32>
  %mat_col = tensor.collapse_shape %mat [[0, 1]] : tensor<100x100xf32> into tensor<10000xf32>
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
