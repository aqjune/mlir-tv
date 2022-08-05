func @olt(%arg0: f32) -> i1 {
  %i = arith.constant 2.0 : f32
  %c = arith.cmpf "olt", %i, %arg0 : f32
  return %c : i1
}
