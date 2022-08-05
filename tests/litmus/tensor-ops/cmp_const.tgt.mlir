func @f() -> tensor<5x3xi1> {
  %c = arith.constant dense<[[1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0]]>: tensor<5x3xi1>
  return %c: tensor<5x3xi1>
}