func @f() -> tensor<?xf32> {
	%c20 = constant 20: index
  %v = linalg.init_tensor [%c20]: tensor<?xf32>
	return %v: tensor<?xf32>
}
