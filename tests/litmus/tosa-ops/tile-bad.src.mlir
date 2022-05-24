// VERIFY-INCORRECT

func.func @f(%x0: tensor<3x3xf32>) -> tensor<6x9xf32> {
  %x = "tosa.reverse"(%x0) {axis = 0: i64}: (tensor<3x3xf32>) -> tensor<3x3xf32>
  %a = "tosa.tile"(%x) {multiples = [2, 1]} : (tensor<3x3xf32>)  -> (tensor<6x3xf32>)
  %b = "tosa.tile"(%a) {multiples = [1, 3]} : (tensor<6x3xf32>)  -> (tensor<6x9xf32>)
  return %b: tensor<6x9xf32>
}
