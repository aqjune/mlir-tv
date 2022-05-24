#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module  {
  func.func @f(%arg0: tensor<5x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> tensor<5x6xf32> {
    %0 = linalg.init_tensor [5, 6] : tensor<5x6xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst: f32) outs(%0: tensor<5x6xf32>) -> tensor<5x6xf32> 
    %cst_0 = arith.constant dense<[1, 0]> : tensor<2xi64>
    %2 = linalg.init_tensor [3, 6] : tensor<3x6xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<6x3xf32>) outs(%2 : tensor<3x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<3x6xf32>
    %4 = linalg.init_tensor [5, 6] : tensor<5x6xf32>
    %5 = linalg.matmul ins(%arg0, %3 : tensor<5x3xf32>, tensor<3x6xf32>) outs(%1 : tensor<5x6xf32>) -> tensor<5x6xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2, %5 : tensor<6xf32>, tensor<5x6xf32>) outs(%4 : tensor<5x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %7 = arith.addf %arg3, %arg4 : f32
      linalg.yield %7 : f32
    } -> tensor<5x6xf32>
    return %6 : tensor<5x6xf32>
  }
}

