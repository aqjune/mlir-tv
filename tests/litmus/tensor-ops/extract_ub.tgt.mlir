func @f() -> ()
{
  %c10 = constant 10 : index
  %v = linalg.init_tensor [%c10]: tensor<?xf32>
  tensor.extract %v[%c10]: tensor<?xf32>
  return
}
