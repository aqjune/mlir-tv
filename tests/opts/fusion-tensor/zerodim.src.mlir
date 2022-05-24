// VERIFY

#map0 = affine_map<() -> ()>

func.func @add_mul_scalar_fusion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
{
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>)
      outs(%0 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
  } -> tensor<f32>
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%1, %arg2 : tensor<f32>, tensor<f32>)
      outs(%0 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %3 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %3 : f32
  } -> tensor<f32>

  return %2 : tensor<f32>
}

// How to reproduce tgt:
// iree-opt -linalg-fusion-for-tensor-ops <src>
