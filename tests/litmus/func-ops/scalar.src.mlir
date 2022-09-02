// VERIFY

func.func private @simpl(%v: f32) -> f32
func.func private @simpl_i(%v: i32) -> i32
func.func private @simpl_tensor(%v: f32) -> tensor<3x5xf32>
func.func private @simpl_large_tensor(%v: i32) -> tensor<4x2x33x55xf32>

func.func @commut(%v1: f32, %v2: f32) -> f32 {
  %r1 = func.call @simpl(%v1): (f32) -> f32
  %r2 = func.call @simpl(%v2): (f32) -> f32
  %r = arith.addf %r1, %r2: f32
  return %r: f32
}

func.func @commut_i(%v1: i32, %v2: i32) -> i32 {
  %r1 = func.call @simpl_i(%v1): (i32) -> i32
  %r2 = func.call @simpl_i(%v2): (i32) -> i32
  %r = arith.addi %r1, %r2: i32
  return %r: i32
}

func.func @commut_tensor(%v1: f32, %v2: f32) -> tensor<3x5xf32> {
  %r1 = func.call @simpl_tensor(%v1): (f32) -> tensor<3x5xf32>
  %r2 = func.call @simpl_tensor(%v2): (f32) -> tensor<3x5xf32>
  %r = "tosa.add"(%r1, %r2) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %r: tensor<3x5xf32>
}

func.func @commut_large_tensor(%v1: i32, %v2: i32) -> tensor<4x2x33x55xf32> {
  %r1 = func.call @simpl_large_tensor(%v1): (i32) -> tensor<4x2x33x55xf32>
  %r2 = func.call @simpl_large_tensor(%v2): (i32) -> tensor<4x2x33x55xf32>
  %r = "tosa.add"(%r1, %r2) : (tensor<4x2x33x55xf32>, tensor<4x2x33x55xf32>) -> tensor<4x2x33x55xf32>
  return %r: tensor<4x2x33x55xf32>
}

func.func @identical_operand(%v1: f32, %v2: f32) -> f32 {
  %v = arith.addf %v1, %v2: f32
  %r = func.call @simpl(%v): (f32) -> f32
  return %r: f32
}

func.func @identical_operand_i(%v1: i32, %v2: i32) -> i32 {
  %v = arith.addi %v1, %v2: i32
  %r = func.call @simpl_i(%v): (i32) -> i32
  return %r: i32
}

func.func @identical_operand_tensor(%v1: f32, %v2: f32) -> tensor<3x5xf32> {
  %v = arith.addf %v1, %v2: f32
  %r = func.call @simpl_tensor(%v): (f32) -> tensor<3x5xf32>
  return %r: tensor<3x5xf32>
}
