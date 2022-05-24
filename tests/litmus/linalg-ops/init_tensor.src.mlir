// EXPECT: "Return value mismatch"

func.func @f() -> index {
	%c0 = arith.constant 0: index
	%c10 = arith.constant 10: index
  %v = linalg.init_tensor [%c10]: tensor<?xf32>
	%d = tensor.dim %v, %c0: tensor<?xf32>
	return %d: index
}
