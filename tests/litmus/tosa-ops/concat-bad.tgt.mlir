func @f(%t1: tensor<1x2xf32>, %t2: tensor<3x2xf32>, %t3: tensor<5x2xf32>) -> tensor<9x2xf32> {
  %res = "tosa.concat"(%t3, %t2, %t1) { axis = 0 : i64}
    : (tensor<5x2xf32>, tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<9x2xf32>
  return %res: tensor<9x2xf32>
}
