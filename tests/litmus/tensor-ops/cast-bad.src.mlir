// VERIFY-INCORRECT

func.func @f(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {
  return %x: tensor<?x?xf32>
}
