func.func @oeq(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ueq", %arg0, %arg1 : f32
  return %c : i1
}