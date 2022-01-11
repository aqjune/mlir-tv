

func @f(%t1: tensor<5xf32>, %t2: tensor<5xf32>) -> f32 {
  %t = "tosa.concat"(%t1, %t2) {axis = 0: i64}: (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
  %identity = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill(%identity, %i) : f32, tensor<f32> -> tensor<f32>

  %rt = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%t : tensor<10xf32>) outs(%outty : tensor<f32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<f32>
  %ret = tensor.extract %rt[] : tensor<f32>
  return %ret : f32
}
