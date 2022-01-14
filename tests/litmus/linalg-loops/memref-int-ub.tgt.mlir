#map0 = affine_map<(i) -> (i)>
#access = [#map0]
#trait = {
  iterator_types = ["parallel"],
  indexing_maps = #access
}
func @f(%m : memref<2xi32>){
  %cst0 = arith.constant 0: index
  %cst1 = arith.constant 1: index
  %i1 = arith.constant 1: i32
  %v1 = memref.load %m[%cst0]: memref<2xi32>
  %v2 = memref.load %m[%cst1]: memref<2xi32>
  %added = arith.addi %v2, %i1: i32
  memref.store %added, %m[%cst1]: memref<2xi32>
  return
}
