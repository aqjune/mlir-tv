// VERIFY

func @f() -> tensor<f32> {
  %a0 = arith.constant -12.0 : f32
  %a1 = arith.constant 3.0 : f32
  %a2 = arith.constant 2.0 : f32
  %a3 = arith.constant 5.0 : f32
  %a4 = arith.constant 4.0 : f32
  %b0 = arith.constant 1.0 : f32
  %b1 = arith.constant 8.0 : f32
  %b2 = arith.constant 5.0 : f32
  %b3 = arith.constant 6.0 : f32
  %b4 = arith.constant 0.0 : f32
  %c0 = arith.mulf %a0, %b0 : f32
  %c1 = arith.mulf %a1, %b1 : f32
  %c2 = arith.mulf %a2, %b2 : f32
  %c3 = arith.mulf %a3, %b3 : f32
  %c4 = arith.mulf %a4, %b4 : f32
  %r1 = arith.addf %c0, %c1 : f32
  %r2 = arith.addf %r1, %c2 : f32
  %r3 = arith.addf %r2, %c3 : f32
  %r4 = arith.addf %r3, %c4 : f32
  %res = linalg.init_tensor []: tensor<f32>
  %res2 = linalg.fill (%r4, %res): f32, tensor<f32> -> tensor<f32>
  return %res2 : tensor<f32>
}
