// VERIFY

func @f() -> f32
{
  %c0 = constant 0 : index
  %cst = constant dense<42.0> : tensor<5xf32>
	%elem = tensor.extract %cst[%c0]: tensor<5xf32>
	return %elem: f32
}
