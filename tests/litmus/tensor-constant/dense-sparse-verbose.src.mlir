// VERIFY

func.func @f() -> tensor<5x3xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0],
       [3.0, 4.0, 5.0],
       [6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0],
       [12.0, 13.0, 14.0]]>: tensor<5x3xf32>
  return %cst: tensor<5x3xf32>
}
