// VERIFY-INCORRECT

func.func @f(%arg0: tensor<10x10xi32>) -> tensor<10x10xi32> {
  %cst = arith.constant 1 : i32
  %init_tensor = tensor.empty () : tensor<10x10xi32>
  %filled = linalg.fill ins(%cst: i32) outs(%init_tensor: tensor<10x10xi32>) -> tensor<10x10xi32>
  %res = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%filled : tensor<10x10xi32>) {
  ^bb0(%arg1: i32):  // no predecessors
    %cst2 = arith.constant 2 : i32
    %three = arith.addi %arg1, %cst2 : i32
    linalg.yield %three : i32
  } -> tensor<10x10xi32>
  return %res : tensor<10x10xi32>
}
