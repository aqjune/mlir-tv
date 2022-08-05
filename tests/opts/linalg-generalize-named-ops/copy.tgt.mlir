#map = affine_map<(d0) -> (d0)>
module  {
  func @copy(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf32>) outs(%arg1 : memref<?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    }
    return
  }
}

