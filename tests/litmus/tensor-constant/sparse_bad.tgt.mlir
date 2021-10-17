func @f() -> f32 {
    %plus_twelve = arith.constant 1.200000e+01 : f32
    return %plus_twelve : f32
}
