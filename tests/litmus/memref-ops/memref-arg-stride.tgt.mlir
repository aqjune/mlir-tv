#map = affine_map<(d0)[s0] -> (d0 + s0)>
module  {
  func @fill_view(%arg0: memref<?xf32, #map>, %arg1: f32) {
    linalg.fill ins(%arg1: f32) outs(%arg0: memref<?xf32, #map>)
    return
  }
}
