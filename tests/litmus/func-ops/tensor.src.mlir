// VERIFY

func.func private @simpl_tensor(%v: f32) -> tensor<3x5xf32>
func.func private @simpl_large_tensor(%v: i32) -> tensor<4x2x33x55xf32>
func.func private @argtensor(%v: tensor<3x5xi32>, %w: i32) -> tensor<3x5xi32>
func.func private @dynamic_tensor(%v: f32) -> tensor<3x5x?xf32>
func.func private @linear_tensor(%t: tensor<5xf32>) -> f32

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

func.func @commut_dynamic_tensor(%v1: f32, %v2: f32) -> tensor<3x5x?xf32> {
  %r1 = func.call @dynamic_tensor(%v1): (f32) -> tensor<3x5x?xf32>
  %r2 = func.call @dynamic_tensor(%v2): (f32) -> tensor<3x5x?xf32>
  %r = "tosa.add"(%r1, %r2) : (tensor<3x5x?xf32>, tensor<3x5x?xf32>) -> tensor<3x5x?xf32>
  return %r: tensor<3x5x?xf32>
}

func.func @argtensor_identical(%v1: tensor<3x5xi32>, %w: i32) -> tensor<3x5xi32> {
  %r = func.call @argtensor(%v1, %w): (tensor<3x5xi32>, i32) -> tensor<3x5xi32>
  return %r: tensor<3x5xi32>
}

func.func @identical_operand_tensor(%v1: f32, %v2: f32) -> tensor<3x5xf32> {
  %v = arith.addf %v1, %v2: f32
  %r = func.call @simpl_tensor(%v): (f32) -> tensor<3x5xf32>
  return %r: tensor<3x5xf32>
}

func.func @slice_tensor(%v: tensor<6xf32>) -> f32 {
  %i = arith.constant 5: index
  %c = arith.constant 5.0: f32
  %mv = tensor.insert %c into %v[%i]: tensor<6xf32>
  %sv = tensor.extract_slice %mv[0][5][1]: tensor<6xf32> to tensor<5xf32>
  %r = func.call @linear_tensor(%sv): (tensor<5xf32>) -> f32
  return %r: f32
}
