func.func @f(%x: tensor<3x3xf32>) -> tensor<6x9xf32> {
  %a = "tosa.tile"(%x) {multiples = [2, 3]} : (tensor<3x3xf32>)  -> (tensor<6x9xf32>)
  return %a: tensor<6x9xf32>
}
