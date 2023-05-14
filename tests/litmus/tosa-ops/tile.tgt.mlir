func.func @f(%x: tensor<3x3xf32>) -> tensor<6x9xf32> {
  %a = "tosa.tile"(%x) {multiples = array<i64: 2, 3>} : (tensor<3x3xf32>)  -> (tensor<6x9xf32>)
  return %a: tensor<6x9xf32>
}
