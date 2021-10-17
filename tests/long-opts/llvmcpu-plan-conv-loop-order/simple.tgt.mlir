#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d3, d2 * 2 + d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
module  {
  func @conv(%arg0: memref<3x3x3x32xf32>, %arg1: memref<1x225x225x3xf32>,
              %arg2: memref<1x112x112x32xf32>) {
    linalg.generic {indexing_maps = [#map0, #map1, #map2],
          iterator_types = ["parallel", "parallel", "parallel",
                            "window", "window", "reduction", "parallel"]}
          ins(%arg0, %arg1 : memref<3x3x3x32xf32>, memref<1x225x225x3xf32>)
          outs(%arg2 : memref<1x112x112x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %0 = arith.mulf %arg3, %arg4 : f32
      %1 = arith.addf %0, %arg5 : f32
      linalg.yield %1 : f32
    }
    return
  }
}
