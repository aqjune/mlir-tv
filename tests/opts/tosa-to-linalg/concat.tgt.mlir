module  {
  func @f(%arg0: tensor<1x2xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<5x2xf32>) -> tensor<9x2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %0 = tensor.dim %arg0, %c0_1 : tensor<1x2xf32>
    %c1_2 = arith.constant 1 : index
    %1 = tensor.dim %arg0, %c1_2 : tensor<1x2xf32>
    %2 = tensor.dim %arg1, %c0 : tensor<3x2xf32>
    %3 = arith.addi %0, %2 : index
    %4 = tensor.dim %arg2, %c0 : tensor<5x2xf32>
    %5 = arith.addi %3, %4 : index
    %6 = linalg.init_tensor [9, 2] : tensor<9x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill(%cst, %6) : f32, tensor<9x2xf32> -> tensor<9x2xf32> 
    %8 = tensor.dim %arg0, %c0 : tensor<1x2xf32>
    %9 = tensor.insert_slice %arg0 into %7[%c0_0, %c0_0] [%8, %1] [%c1, %c1] : tensor<1x2xf32> into tensor<9x2xf32>
    %10 = arith.addi %c0_0, %8 : index
    %11 = tensor.dim %arg1, %c0 : tensor<3x2xf32>
    %12 = tensor.insert_slice %arg1 into %9[%10, %c0_0] [%11, %1] [%c1, %c1] : tensor<3x2xf32> into tensor<9x2xf32>
    %13 = arith.addi %10, %11 : index
    %14 = tensor.dim %arg2, %c0 : tensor<5x2xf32>
    %15 = tensor.insert_slice %arg2 into %12[%13, %c0_0] [%14, %1] [%c1, %c1] : tensor<5x2xf32> into tensor<9x2xf32>
    %16 = arith.addi %13, %14 : index
    return %15 : tensor<9x2xf32>
  }
}

