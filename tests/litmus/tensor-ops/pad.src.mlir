// VERIFY

func.func @f(%t: tensor<?x?xf32>, %pad_value: f32) -> (f32, f32, f32, f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %d0 = tensor.dim %t, %c0: tensor<?x?xf32>
  %d1 = tensor.dim %t, %c1: tensor<?x?xf32>

  %res = tensor.pad %t low[1, 2] high[2, 3] {
  ^bb0(%arg0 : index, %arg1 : index):
    tensor.yield %pad_value : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>

  %y = arith.addi %d0, %c2: index
  %x = arith.addi %d1, %c4: index
  %p1 = tensor.extract %res[%c1, %c0]: tensor<?x?xf32>
  %p2 = tensor.extract %res[%d0, %c1]: tensor<?x?xf32>
  %p3 = tensor.extract %res[%y, %c0]: tensor<?x?xf32>
  %p4 = tensor.extract %res[%c1, %x]: tensor<?x?xf32>
  %p5 = tensor.extract %res[%y, %x]: tensor<?x?xf32>
  return %p1, %p2, %p3, %p4, %p5: f32,f32,f32,f32,f32
}
