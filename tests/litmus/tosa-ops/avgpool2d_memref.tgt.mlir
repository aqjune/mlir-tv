#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @avgpool(%arg0: tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32> {
  %c49_i32 = arith.constant 49 : i32
  %0 = bufferization.to_memref %arg0 : tensor<1x7x7x1280xf32> to memref<1x7x7x1280xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = memref.alloc() : memref<1x1x1x1280xf32>
  linalg.fill ins(%cst: f32) outs(%1: memref<1x1x1x1280xf32>)
  %2 = memref.alloc() : memref<7x7xf32>
  linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%0, %2 : memref<1x7x7x1280xf32>, memref<7x7xf32>) outs(%1 : memref<1x1x1x1280xf32>)
  %3 = memref.alloc() : memref<1x1x1x1280xf32>
  linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : memref<1x1x1x1280xf32>) outs(%3 : memref<1x1x1x1280xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %615 = arith.sitofp %c49_i32 : i32 to f32
    %616 = arith.divf %arg1, %615 : f32
    linalg.yield %616 : f32
  }
  %4 = bufferization.to_tensor %3 : memref<1x1x1x1280xf32> to tensor<1x1x1x1280xf32>
  return %4 : tensor<1x1x1x1280xf32>
}
