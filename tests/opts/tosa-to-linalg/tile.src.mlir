// VERIFY

func.func @f(%x: tensor<3x3xf32>) -> tensor<6x9xf32> {
  %a = "tosa.tile"(%x) {multiples = array<i64: 2, 1>} : (tensor<3x3xf32>)  -> (tensor<6x3xf32>)
  %b = "tosa.tile"(%a) {multiples = array<i64: 1, 3>} : (tensor<6x3xf32>)  -> (tensor<6x9xf32>)
  return %b: tensor<6x9xf32>
}

// mlir-opt -tosa-to-linalg
