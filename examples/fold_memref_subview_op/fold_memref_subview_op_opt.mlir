#map = affine_map<(d0)[s0] -> (d0 + s0)>
module  {
  func @fold_subview(%arg0: tensor<64x64xf32>, %arg1: index, %arg2: index) -> f32 {
    %c0 = constant 0 : index
    %0 = memref.buffer_cast %arg0 : memref<64x64xf32>
    %1 = affine.apply #map(%c0)[%arg1]
    %2 = memref.load %0[%1, %arg2] : memref<64x64xf32>
    return %2 : f32
  }
}

