// VERIFY

func.func @eq(%arg0: f32, %arg1: f32) -> i1 {
  %e = arith.cmpf "oeq", %arg0, %arg1 : f32
  return %e: i1
}

func.func @ne(%arg0: f32, %arg1: f32) -> i1 {
  %e = arith.cmpf "one", %arg0, %arg1 : f32
  return %e: i1
}

func.func @lt(%arg0: f32, %arg1: f32) -> i1 {
  %e = arith.cmpf "olt", %arg0, %arg1 : f32
  return %e: i1
}

func.func @gt(%arg0: f32, %arg1: f32) -> i1 {
  %e = arith.cmpf "ogt", %arg0, %arg1 : f32
  return %e: i1
}
