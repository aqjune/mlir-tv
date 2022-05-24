func.func @oeq(%arg0: f32) -> i1 {
  %i = arith.constant -0.0 : f32
  %c = arith.cmpf "oeq", %i, %arg0 : f32
  return %c : i1
}

func.func @one(%arg0: f32) -> i1 {
  %i = arith.constant -0.0 : f32
  %c = arith.cmpf "one", %i, %arg0 : f32
  return %c : i1
}

func.func @ueq(%arg0: f32) -> i1 {
  %i = arith.constant -0.0 : f32
  %c = arith.cmpf "ueq", %i, %arg0 : f32
  return %c : i1
}

func.func @une(%arg0: f32) -> i1 {
  %i = arith.constant -0.0 : f32
  %c = arith.cmpf "une", %i, %arg0 : f32
  return %c : i1
}
