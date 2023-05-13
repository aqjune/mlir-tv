func.func @f() -> tensor<?xf32> {
	%c20 = arith.constant 20: index
  %v = tensor.empty (%c20): tensor<?xf32>
	return %v: tensor<?xf32>
}
