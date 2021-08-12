func @f(%arg0: index) -> (f32, f32) {
    %cst_0 = constant 4.000000e+00 : f32
    %cst_1 = constant -2.000000e+00 : f32
    return %cst_0, %cst_1 : f32, f32
}
