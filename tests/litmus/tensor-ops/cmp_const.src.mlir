// VERIFY

func @f() -> tensor<5x3xi1> {
  %lhs = arith.constant dense<[[0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0],
    [12.0, 13.0, 14.0]]>: tensor<5x3xf32>
  %rhs = arith.constant dense<[[1.0, 0.0, 3.0],
    [2.0, 5.0, 4.0],
    [7.0, 6.0, 9.0],
    [8.0, 11.0, 10.0],
    [13.0, 12.0, 14.0]]>: tensor<5x3xf32>
  %c = arith.cmpf "olt", %lhs, %rhs : tensor<5x3xf32>
  return %c: tensor<5x3xi1>
}