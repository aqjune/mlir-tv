// VERIFY

func @f() -> i8
{
  %c0 = constant 0 : index
  %cst = constant dense<42> : tensor<5xi8>
	%elem = tensor.extract %cst[%c0]: tensor<5xi8>
	return %elem: i8
}
