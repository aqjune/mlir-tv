// VERIFY

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (0)>
func.func @consumer_with_reduction(%arg0: tensor<1x10xf32>,
                              %arg1: tensor<1x10xf32>,
                              %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %init = linalg.init_tensor [1, 10] : tensor<1x10xf32>
  %0 = linalg.generic
    {indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x10xf32>, tensor<1x10xf32>)
    outs(%init : tensor<1x10xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1x10xf32>
  %1 = linalg.generic
    {indexing_maps = [#map1, #map2],
     iterator_types = ["reduction"]}
    ins(%0 : tensor<1x10xf32>)
    outs(%arg2 : tensor<1xf32>)  {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

