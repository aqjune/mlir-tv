func @f(%arg0: tensor<10x10xi32>) -> tensor<10x10xi32> {
  %not_three= arith.constant 4 : i32
  %init_tensor = linalg.init_tensor [10, 10] : tensor<10x10xi32>
  %filled = linalg.fill(%not_three, %init_tensor): i32, tensor<10x10xi32> -> tensor<10x10xi32>
  return %filled : tensor<10x10xi32>
}
