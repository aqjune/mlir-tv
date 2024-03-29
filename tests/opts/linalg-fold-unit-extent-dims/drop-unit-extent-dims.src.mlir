// VERIFY

func.func @unit_dim_for_both_reduction(%arg0: tensor<1x?x1x1xf32>) -> tensor<1x1xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %1 = tensor.empty () : tensor<1x1xf32>
  %2 = linalg.fill ins(%cst: f32) outs(%1: tensor<1x1xf32>) -> tensor<1x1xf32>
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0 : tensor<1x?x1x1xf32>)
    outs(%2 : tensor<1x1xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  
    %4 = arith.addf %arg1, %arg2 : f32
    linalg.yield %4 : f32
  } -> tensor<1x1xf32>
  return %3 : tensor<1x1xf32>
}

// How to reproduce tgt:
// mlir-opt -pass-pipeline="func(linalg-fold-unit-extent-dims)" <src>
