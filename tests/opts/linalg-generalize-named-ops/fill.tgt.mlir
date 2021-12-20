#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func @generalize_fill(%arg0: memref<20x20xf32>, %arg1: f32) {
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : f32) outs(%arg0 : memref<20x20xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    }
    return
  }
}
