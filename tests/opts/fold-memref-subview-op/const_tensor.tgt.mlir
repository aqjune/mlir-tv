#map = affine_map<(d0)[s0] -> (d0 + s0)>
module  {
  func.func @fold_subview(%arg0: tensor<64x64xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> f32 {
    %0 = bufferization.to_memref %arg0 : tensor<64x64xf32> to memref<64x64xf32>
    %1 = affine.apply #map(%arg5)[%arg1]
    %2 = affine.apply #map(%arg6)[%arg2]
    %3 = memref.load %0[%1, %2] : memref<64x64xf32>
    return %3 : f32
  }
}

