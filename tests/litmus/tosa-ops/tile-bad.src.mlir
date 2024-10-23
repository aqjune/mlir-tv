// VERIFY-INCORRECT

func.func @f(%x0: tensor<3x3xf32>) -> tensor<6x9xf32> {
  %x = "tosa.reverse"(%x0) {axis = 0: i32}: (tensor<3x3xf32>) -> tensor<3x3xf32>
  %a = "tosa.tile"(%x) {multiples = array<i64: 2, 1>} : (tensor<3x3xf32>)  -> (tensor<6x3xf32>)
  %b = "tosa.tile"(%a) {multiples = array<i64: 1, 3>} : (tensor<6x3xf32>)  -> (tensor<6x9xf32>)
  return %b: tensor<6x9xf32>
}
