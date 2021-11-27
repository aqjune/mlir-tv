// VERIFY-INCORRECT

#map0 = affine_map<(i) -> (i)>
#access = [#map0, #map0]
#trait = {
  iterator_types = ["parallel"],
  indexing_maps = #access
}
func @f(%arg0 : memref<2xi32>){
  linalg.generic #trait
     ins(%arg0 : memref<2xi32>)
    outs(%arg0 : memref<2xi32>) {
       ^bb0(%arg3: i32, %arg4: i32) :
    %idx0 = linalg.index 0 : index
    %idx = arith.index_cast %idx0 : index to i32
    %added = arith.addi %idx, %arg3 : i32
    linalg.yield %added : i32
  }
  return
}
