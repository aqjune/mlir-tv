// sum (sum(A), sum(B)) → sum(A::B)

func @f(%t1: tensor<10xf32>, %t2: tensor<10xf32>) -> f32 {
  %identity = arith.constant -0.0 : f32
  %i1 = linalg.init_tensor []: tensor<f32>
  %i2 = linalg.init_tensor []: tensor<f32>
  %outty1 = linalg.fill(%identity, %i1) : f32, tensor<f32> -> tensor<f32>
  %outty2 = linalg.fill(%identity, %i2) : f32, tensor<f32> -> tensor<f32>

  %rt1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%t1 : tensor<10xf32>) outs(%outty1 : tensor<f32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<f32>
  %ret1 = tensor.extract %rt1[] : tensor<f32>

  %rt2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%t2 : tensor<10xf32>) outs(%outty2 : tensor<f32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<f32>
  %ret2 = tensor.extract %rt2[] : tensor<f32>

  %ret = arith.addf %ret1, %ret2 : f32
  return %ret : f32
}
