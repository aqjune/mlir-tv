// VERIFY-INCORRECT

// Introducing memref.load %a is incorrect because %a may not be dereferenceable

func @f(%a: memref<1xf32>) {
  return
}
