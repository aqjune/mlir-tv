// VERIFY

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @producer_indexed_consumer_fusion(%arg0: tensor<?x?xi32>,
                                       %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xi32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1  : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):       // no predecessors
      %10 = addi %arg2, %arg3 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
  %4 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%3 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):       // no predecessors
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %5 = index_cast %idx0 : index to i32
      %6 = index_cast %idx1 : index to i32
      %7 = addi %arg2, %5 : i32
      %8 = subi %7, %6 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}

// How to reproduce tgt:
// iree-opt -linalg-fusion-for-tensor-ops <src>
