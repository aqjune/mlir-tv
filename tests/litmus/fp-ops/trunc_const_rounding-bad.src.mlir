// VERIFY-INCORRECT

func.func @round() -> i1 {
  %c = arith.constant 3.00000012 : f64
  %ct = arith.truncf %c: f64 to f32
  %f = arith.constant 3.0000004 : f64
  %ft = arith.truncf %f: f64 to f32
  %e = arith.cmpf "oeq", %ct, %ft : f32
  return %e: i1
}
