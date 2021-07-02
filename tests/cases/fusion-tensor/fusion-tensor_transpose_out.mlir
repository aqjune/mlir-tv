#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
module  {
  func @transpose_add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = memref.dim %arg0, %c0 : tensor<?x?xf32>
    %1 = memref.dim %arg0, %c1 : tensor<?x?xf32>
    %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %4 = addf %arg3, %arg4 : f32
      %5 = mulf %4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
  }
}
