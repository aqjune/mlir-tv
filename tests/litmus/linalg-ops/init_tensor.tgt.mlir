func @f() -> index {
	%c0 = constant 0: index
	%c10 = constant 20: index
  %v = linalg.init_tensor [%c10]: tensor<?xf32>
	%d = tensor.dim %v, %c0: tensor<?xf32>
	return %d: index
}
