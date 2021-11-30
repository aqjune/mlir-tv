func @f(%a: memref<1xf32>) {
  %c0 = arith.constant 0: index
  %v = memref.load %a[%c0]: memref<1xf32>
  return
}
