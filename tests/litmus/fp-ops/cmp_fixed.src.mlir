// VERIFY

func.func @true(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "true", %arg0, %arg1 : f32
  return %c : i1
}

func.func @false(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "false", %arg0, %arg1 : f32
  return %c : i1
}
