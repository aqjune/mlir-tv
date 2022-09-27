func.func @i32_to_i64() -> i64 {
  %c = arith.constant 2147483647: i64
  return %c: i64
}

func.func @i16_to_i64() -> i64 {
  %c = arith.constant 32767: i64
  return %c: i64
}

func.func @neg_i32_to_i64() -> i64 {
  %c = arith.constant 0xffffffff80000000: i64
  return %c: i64
}

func.func @tensor_i32_to_i64() -> tensor<5xi64> {
  %x = arith.constant dense<2147483647> : tensor<5xi64>
  return %x: tensor<5xi64>
}

func.func @tensor_neg_i32_to_i64() -> tensor<5xi64> {
  %x = arith.constant dense<0xffffffff80000000> : tensor<5xi64>
  return %x: tensor<5xi64>
}
