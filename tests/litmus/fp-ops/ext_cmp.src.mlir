// VERIFY

func.func @eq(%arg0: f32, %arg1: f32) -> i1 {
  %lhs = arith.extf %arg0: f32 to f64
  %rhs = arith.extf %arg1: f32 to f64
  %e = arith.cmpf "oeq", %lhs, %rhs : f64
  return %e: i1
}

func.func @ne(%arg0: f32, %arg1: f32) -> i1 {
  %lhs = arith.extf %arg0: f32 to f64
  %rhs = arith.extf %arg1: f32 to f64
  %e = arith.cmpf "one", %lhs, %rhs : f64
  return %e: i1
}

func.func @lt(%arg0: f32, %arg1: f32) -> i1 {
  %lhs = arith.extf %arg0: f32 to f64
  %rhs = arith.extf %arg1: f32 to f64
  %e = arith.cmpf "olt", %lhs, %rhs : f64
  return %e: i1
}

func.func @gt(%arg0: f32, %arg1: f32) -> i1 {
  %lhs = arith.extf %arg0: f32 to f64
  %rhs = arith.extf %arg1: f32 to f64
  %e = arith.cmpf "ogt", %lhs, %rhs : f64
  return %e: i1
}
