// EXPECT: "correct (source is always undefined)"

func.func @f() -> ()
{
  %c10 = arith.constant 10 : index
  %v = tensor.empty (%c10): tensor<?xf32>
  tensor.extract %v[%c10]: tensor<?xf32>
  return
}
