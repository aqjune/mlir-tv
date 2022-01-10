

// sum (reverse(variable)) = sum(variable)

// Size 100
// Z3 hash : 1 msec
// CVC5 hash : : 1 msec
// Z3 multiset : timeout
// CVC5 multiset : 4 sec

func @f(%t: tensor<100xf32>) -> tensor<f32> {
  %rt = "tosa.reverse"(%t) {axis = 0 : i64} : (tensor<100xf32>) -> tensor<100xf32>
  %identity = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill(%identity, %i) : f32, tensor<f32> -> tensor<f32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%rt : tensor<100xf32>) outs(%outty : tensor<f32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
    %0 = arith.addf %arg0, %arg1 : f32
    linalg.yield %0 : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}
