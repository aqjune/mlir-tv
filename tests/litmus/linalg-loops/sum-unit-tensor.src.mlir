// VERIFY

func @sum(%x: tensor<1xf32>) -> f32
{
  %zero = arith.constant -0.0 : f32
  %i = linalg.init_tensor [] : tensor<f32>
  %outty = linalg.fill(%zero, %i): f32, tensor<f32> -> tensor<f32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
    ins(%x : tensor<1xf32>) outs(%outty : tensor<f32>) {
    ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
  } -> tensor<f32>
  %sum = tensor.extract %result[] : tensor<f32>
  return %sum : f32
}
