// VERIFY
// ARGS: --associative

func.func @f() -> f32 {
  %a0 = arith.constant -12.0 : f32
  %a1 = arith.constant 2.0 : f32
  %a2 = arith.constant 3.0 : f32
  %a3 = arith.constant 0.0 : f32
  %c0 = arith.addf %a0, %a1 : f32
  %c1 = arith.addf %a2, %a3 : f32
  %sum = arith.addf %c0, %c1: f32
  return %sum: f32
}
