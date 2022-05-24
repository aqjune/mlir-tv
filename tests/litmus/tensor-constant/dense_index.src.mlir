// VERIFY

func.func @f() -> index
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<42> : tensor<5xindex>
	%elem = tensor.extract %cst[%c0]: tensor<5xindex>
	return %elem: index
}
