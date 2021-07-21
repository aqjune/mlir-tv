// EXPECT: "Return value mismatch"

func @f() -> tensor<?xf32> {
	%c10 = constant 10: index
  %v = linalg.init_tensor [%c10]: tensor<?xf32>
	return %v: tensor<?xf32>
}
