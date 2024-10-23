// VERIFY-INCORRECT

func.func @f(%t1: tensor<1x2xf32>, %t2: tensor<3x2xf32>, %t3: tensor<5x2xf32>) -> tensor<9x2xf32> {
  %res = "tosa.concat"(%t1, %t2, %t3) { axis = 0 : i32}
    : (tensor<1x2xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<9x2xf32>
  return %res: tensor<9x2xf32>
}
