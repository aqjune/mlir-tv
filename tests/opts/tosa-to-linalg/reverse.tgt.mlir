#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module  {
  func @f(%arg0: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xi32>
    %c1 = arith.constant 1 : index
    %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xi32>
    %c2 = arith.constant 2 : index
    %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xi32>
    %c1_0 = arith.constant 1 : index
    %3 = tensor.dim %arg0, %c1_0 : tensor<?x?x?xi32>
    %4 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xi32>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : tensor<?x?x?xi32>) {
    ^bb0(%arg1: i32):  // no predecessors
      %6 = linalg.index 0 : index
      %7 = linalg.index 1 : index
      %c1_1 = arith.constant 1 : index
      %8 = arith.subi %3, %c1_1 : index
      %9 = arith.subi %8, %7 : index
      %10 = linalg.index 2 : index
      %11 = tensor.extract %arg0[%6, %9, %10] : tensor<?x?x?xi32>
      linalg.yield %11 : i32
    } -> tensor<?x?x?xi32>
    return %5 : tensor<?x?x?xi32>
  }
}

