// EXPECT: "Return value mismatch"
// SKIP-IDCHECK

func.func @f() -> tensor<?xf32> {
	%c10 = arith.constant 10: index
  %v = tensor.empty (%c10): tensor<?xf32>
	return %v: tensor<?xf32>
}
