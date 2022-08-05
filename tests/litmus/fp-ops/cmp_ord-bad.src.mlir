// VERIFY-INCORRECT

func @oeq(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "oeq", %arg0, %arg1 : f32
  return %c : i1
}