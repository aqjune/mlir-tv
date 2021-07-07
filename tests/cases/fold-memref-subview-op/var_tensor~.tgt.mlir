#map = affine_map<(d0)[s0] -> (d0 + s0)>
module  {
  func @fold_subview(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> f32 {
    %0 = memref.buffer_cast %arg0 : memref<?x?xf32>
    %1 = affine.apply #map(%arg5)[%arg1]
    %2 = affine.apply #map(%arg6)[%arg2]
    %3 = memref.load %0[%2, %1] : memref<?x?xf32> // different mapping
    return %3 : f32
  }
}

