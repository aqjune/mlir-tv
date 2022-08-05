func @f() -> tensor<3x5xf32> {
  %cst = arith.constant dense<[[0.0, 3.0, 6.0, 9.0, 12.0],
      [1.0, 4.0, 7.0, 10.0, 13.0],
      [2.0, 5.0, 8.0, 11.0, 14.0]]>: tensor<3x5xf32>
  return %cst: tensor<3x5xf32>
}
