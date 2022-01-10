func @f(%t: tensor<10x20xf32>) -> tensor<f32> {
  %identity = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill(%identity, %i) : f32, tensor<f32> -> tensor<f32>

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                    affine_map<(d0, d1) -> ()>],
    iterator_types = ["reduction", "reduction"]}
    ins(%t : tensor<10x20xf32>) outs(%outty : tensor<f32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
