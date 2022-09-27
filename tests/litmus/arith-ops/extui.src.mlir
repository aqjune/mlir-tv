// VERIFY

func.func @i32_to_i64() -> i64 {
  %c = arith.constant 2147483647: i32
  %x = arith.extui %c: i32 to i64
  return %x: i64
}

func.func @i16_to_i64() -> i64 {
  %c = arith.constant 32767: i16
  %x = arith.extui %c: i16 to i64
  return %x: i64
}

func.func @neg_i32_to_i64() -> i64 {
  %c = arith.constant 2147483648: i32
  %x = arith.extui %c: i32 to i64
  return %x: i64
}

func.func @tensor_i32_to_i64() -> tensor<5xi64> {
  %c = arith.constant dense<2147483647> : tensor<5xi32>
  %x = arith.extui %c: tensor<5xi32> to tensor<5xi64>
  return %x: tensor<5xi64>
}

func.func @tensor_neg_i32_to_i64() -> tensor<5xi64> {
  %c = arith.constant dense<2147483648> : tensor<5xi32>
  %x = arith.extui %c: tensor<5xi32> to tensor<5xi64>
  return %x: tensor<5xi64>
}
