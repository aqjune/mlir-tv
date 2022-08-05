// UNSUPPORTED

// To support this pair, sum(x1, x2, ..) = x1 + x2 + .. must be supported.

func @f(%arg0: tensor<1x?x1x1xi32>) -> tensor<1x1xi32> {
  %cst = arith.constant 1 : i32
  %init_tensor = linalg.init_tensor [1, 1] : tensor<1x1xi32>
  %filled = linalg.fill(%cst, %init_tensor): i32, tensor<1x1xi32> -> tensor<1x1xi32>
  %res = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0 : tensor<1x?x1x1xi32>)
    outs(%filled : tensor<1x1xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %4 = arith.addi %arg1, %arg2 : i32
    linalg.yield %4 : i32
  } -> tensor<1x1xi32>
  return %res : tensor<1x1xi32>
}

// How to reproduce tgt:
// mlir-opt -linalg-fold-unit-extent-dims <src>
