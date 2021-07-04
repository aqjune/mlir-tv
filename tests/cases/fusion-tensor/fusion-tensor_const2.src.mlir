// VERIFY-INCORRECT

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cst = constant dense<42.0> : tensor<5xf32>
  %0 = memref.dim %arg0, %c1 : tensor<5x?x?xf32>
  %1 = memref.dim %arg0, %c2 : tensor<5x?x?xf32>
  %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%cst, %arg0 : tensor<5xf32>, tensor<5x?x?xf32>)
    outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
  return %3 : tensor<5x?x?xf32>
}

// How to reproduce tgt:
// iree-opt -linalg-fusion-for-tensor-ops <src>
