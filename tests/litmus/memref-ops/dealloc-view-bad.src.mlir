// VERIFY-INCORRECT

// subview ptr cannot be freed
func @f() {
  %ptr = memref.alloc(): memref<8x64xf32>
  return
}
