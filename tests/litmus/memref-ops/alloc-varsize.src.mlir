// EXPECT: "correct (source is always undefined)"

func.func @f(%sz: index) -> (){
  %ptr = memref.alloc(%sz): memref<?xf32>
  memref.load %ptr[%sz]: memref<?xf32>
  return
}
