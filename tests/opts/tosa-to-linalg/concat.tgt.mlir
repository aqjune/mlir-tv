module  {
  func.func @f(%arg0: tensor<1x2xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<5x2xf32>) -> tensor<9x2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c9 = arith.constant 9 : index
    %0 = tensor.empty () : tensor<9x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst: f32) outs(%0: tensor<9x2xf32>) -> tensor<9x2xf32> 
    %c1_4 = arith.constant 1 : index
    %2 = tensor.insert_slice %arg0 into %1[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<9x2xf32>
    %c1_5 = arith.constant 1 : index
    %c3_6 = arith.constant 3 : index
    %3 = tensor.insert_slice %arg1 into %2[1, 0] [3, 2] [1, 1] : tensor<3x2xf32> into tensor<9x2xf32>
    %c4_7 = arith.constant 4 : index
    %c5_8 = arith.constant 5 : index
    %4 = tensor.insert_slice %arg2 into %3[4, 0] [5, 2] [1, 1] : tensor<5x2xf32> into tensor<9x2xf32>
    %c9_9 = arith.constant 9 : index
    return %4 : tensor<9x2xf32>
  }
}

