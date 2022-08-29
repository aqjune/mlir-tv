// VERIFY

func.func private @simpl(%v: f32) -> f32
func.func private @simpl_i(%v: i32) -> i32

func.func @assoc(%v1: f32, %v2: f32) -> f32 {
  %r1 = func.call @simpl(%v1): (f32) -> f32
  %r2 = func.call @simpl(%v2): (f32) -> f32
  %r = arith.addf %r1, %r2: f32
  return %r: f32
}

func.func @assoc_i(%v1: i32, %v2: i32) -> i32 {
  %r1 = func.call @simpl_i(%v1): (i32) -> i32
  %r2 = func.call @simpl_i(%v2): (i32) -> i32
  %r = arith.addi %r1, %r2: i32
  return %r: i32
}

func.func @operand_assoc(%v1: f32, %v2: f32) -> f32 {
  %v = arith.addf %v1, %v2: f32
  %r = func.call @simpl(%v): (f32) -> f32
  return %r: f32
}

func.func @operand_assoc_i(%v1: i32, %v2: i32) -> i32 {
  %v = arith.addi %v1, %v2: i32
  %r = func.call @simpl_i(%v): (i32) -> i32
  return %r: i32
}
