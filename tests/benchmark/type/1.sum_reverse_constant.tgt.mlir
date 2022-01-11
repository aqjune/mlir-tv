
func @f() -> tensor<f32> {
  %t = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]> : tensor<10xf32>
  %identity = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill(%identity, %i) : f32, tensor<f32> -> tensor<f32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%t : tensor<10xf32>) outs(%outty : tensor<f32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
