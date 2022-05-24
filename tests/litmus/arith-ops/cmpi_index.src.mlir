// VERIFY

func.func @f() -> i1 {
  %c1 = arith.constant 1 : index
  %c13 = arith.constant 13 : index
  %0 = arith.cmpi slt, %c13, %c1 : index
  return %0 : i1
}
