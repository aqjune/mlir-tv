func.func @f(%arg0: tensor<10x10xi32>) -> tensor<10x10xi32> {
  %three= arith.constant 3 : i32
  %init_tensor = linalg.init_tensor [10, 10] : tensor<10x10xi32>
  %filled = linalg.fill ins(%three: i32) outs(%init_tensor: tensor<10x10xi32>) -> tensor<10x10xi32>
  return %filled : tensor<10x10xi32>
}
