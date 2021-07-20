// VERIFY-INCORRECT

func @sum(%mat: tensor<5x5xf32>) -> tensor<5xf32>
{
  %outty = linalg.init_tensor [5] : tensor<5xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
     ins(%mat : tensor<5x5xf32>) outs(%outty : tensor<5xf32>) {
     ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
  } -> tensor<5xf32>
  return %result : tensor<5xf32>
}
