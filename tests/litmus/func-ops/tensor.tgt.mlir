func.func private @simpl_tensor(%v: f32) -> tensor<3x5xf32>
func.func private @simpl_large_tensor(%v: i32) -> tensor<4x2x33x55xf32>
func.func private @argtensor(%v: tensor<3x5xi32>, %w: i32) -> tensor<3x5xi32>

func.func @commut_tensor(%v1: f32, %v2: f32) -> tensor<3x5xf32> {
  %r1 = func.call @simpl_tensor(%v1): (f32) -> tensor<3x5xf32>
  %r2 = func.call @simpl_tensor(%v2): (f32) -> tensor<3x5xf32>
  %r = "tosa.add"(%r2, %r1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %r: tensor<3x5xf32>
}

func.func @commut_large_tensor(%v1: i32, %v2: i32) -> tensor<4x2x33x55xf32> {
  %r1 = func.call @simpl_large_tensor(%v1): (i32) -> tensor<4x2x33x55xf32>
  %r2 = func.call @simpl_large_tensor(%v2): (i32) -> tensor<4x2x33x55xf32>
  %r = "tosa.add"(%r2, %r1) : (tensor<4x2x33x55xf32>, tensor<4x2x33x55xf32>) -> tensor<4x2x33x55xf32>
  return %r: tensor<4x2x33x55xf32>
}

func.func @argtensor_identical(%v1: tensor<3x5xi32>, %w: i32) -> tensor<3x5xi32> {
  %r = func.call @argtensor(%v1, %w): (tensor<3x5xi32>, i32) -> tensor<3x5xi32>
  return %r: tensor<3x5xi32>
}

func.func @identical_operand_tensor(%v1: f32, %v2: f32) -> tensor<3x5xf32> {
  %v = arith.addf %v2, %v1: f32
  %r = func.call @simpl_tensor(%v): (f32) -> tensor<3x5xf32>
  return %r: tensor<3x5xf32>
}
