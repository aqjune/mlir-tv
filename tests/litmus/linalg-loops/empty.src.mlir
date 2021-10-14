// UNSUPPORTED

#accesses = [
  affine_map<(i, j) -> (i, j)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"]
}
func @dead_linalg_tensor(%arg0 : tensor<7x7xi32>) {
  %0 = linalg.generic #trait outs(%arg0 : tensor<7x7xi32>) {
  ^bb(%3: i32) :
    linalg.yield %3 : i32
  } -> tensor<7x7xi32>
  return
}
