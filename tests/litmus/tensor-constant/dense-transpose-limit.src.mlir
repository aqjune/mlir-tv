// ARGS: -max-const-tensor-size=3
// VERIFY

func.func @f() -> tensor<3x5xf32> {
  %cst = arith.constant dense<[[0.0, 1.0, 2.0],
       [3.0, 4.0, 5.0],
       [6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0],
       [12.0, 13.0, 14.0]]>: tensor<5x3xf32>
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %tp = "tosa.transpose"(%cst, %perm): (tensor<5x3xf32>, tensor<2xi32>) -> tensor<3x5xf32>
  return %tp: tensor<3x5xf32>
}
