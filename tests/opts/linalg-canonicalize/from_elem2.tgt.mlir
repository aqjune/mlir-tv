module  {
  func.func @from_elem2() -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    %cst = arith.constant 1.100000e+01 : f32
    %cst_0 = arith.constant 1.000000e+01 : f32
    %cst_1 = arith.constant 9.000000e+00 : f32
    %cst_2 = arith.constant 8.000000e+00 : f32
    %cst_3 = arith.constant 7.000000e+00 : f32
    %cst_4 = arith.constant 6.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e+00 : f32
    %cst_6 = arith.constant 4.000000e+00 : f32
    %cst_7 = arith.constant 3.000000e+00 : f32
    %cst_8 = arith.constant 2.000000e+00 : f32
    %cst_9 = arith.constant 1.000000e+00 : f32
    %cst_10 = arith.constant 0.000000e+00 : f32
    return %cst_10, %cst_9, %cst_8, %cst_7, %cst_6, %cst_5, %cst_4, %cst_3, %cst_2, %cst_1, %cst_0, %cst : f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
  }
}

