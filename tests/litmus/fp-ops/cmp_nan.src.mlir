// VERIFY

func @oeq(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "oeq", %i, %arg0 : f32
  return %c : i1
}

func @one(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "one", %i, %arg0 : f32
  return %c : i1
}

func @ole(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "ole", %i, %arg0 : f32
  return %c : i1
}

func @olt(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "olt", %i, %arg0 : f32
  return %c : i1
}

func @oge(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "oge", %i, %arg0 : f32
  return %c : i1
}

func @ogt(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "ogt", %i, %arg0 : f32
  return %c : i1
}

func @ueq(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "ueq", %i, %arg0 : f32
  return %c : i1
}

func @une(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "une", %i, %arg0 : f32
  return %c : i1
}

func @ule(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "ule", %i, %arg0 : f32
  return %c : i1
}

func @ult(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "ult", %i, %arg0 : f32
  return %c : i1
}

func @uge(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "uge", %i, %arg0 : f32
  return %c : i1
}

func @ugt(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "ugt", %i, %arg0 : f32
  return %c : i1
}

func @ord(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "ord", %i, %arg0 : f32
  return %c : i1
}

func @uno(%arg0: f32) -> i1 {
  %i = arith.constant 0x7FC00000 : f32
  %c = arith.cmpf "uno", %i, %arg0 : f32
  return %c : i1
}
