func @true(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.constant 1 : i1
  return %c : i1
}

func @false(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.constant 0 : i1
  return %c : i1
}
