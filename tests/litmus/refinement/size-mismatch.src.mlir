// EXPECT: "Return value mismatch"
// NO-IDENTITY

func @f() -> tensor<?xf32> {
	%c10 = arith.constant 10: index
  %v = linalg.init_tensor [%c10]: tensor<?xf32>
	return %v: tensor<?xf32>
}
