// VERIFY

func.func @oeq() -> i1 {
  %i = arith.constant 3.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "oeq", %i, %v : f32
  return %c : i1
}

func.func @one() -> i1 {
  %i = arith.constant 2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "one", %i, %v : f32
  return %c : i1
}

func.func @ole() -> i1 {
  %i = arith.constant 2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "ole", %i, %v : f32
  return %c : i1
}

func.func @olt() -> i1 {
  %i = arith.constant 2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "olt", %i, %v : f32
  return %c : i1
}

func.func @oge() -> i1 {
  %i = arith.constant 3.0 : f32
  %v = arith.constant 2.0 : f32
  %c = arith.cmpf "oge", %i, %v : f32
  return %c : i1
}

func.func @ogt() -> i1 {
  %i = arith.constant 3.0 : f32
  %v = arith.constant 2.0 : f32
  %c = arith.cmpf "ogt", %i, %v : f32
  return %c : i1
}

func.func @ueq() -> i1 {
  %i = arith.constant 3.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "ueq", %i, %v : f32
  return %c : i1
}

func.func @une() -> i1 {
  %i = arith.constant 2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "une", %i, %v : f32
  return %c : i1
}

func.func @ule() -> i1 {
  %i = arith.constant 2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "ule", %i, %v : f32
  return %c : i1
}

func.func @ult() -> i1 {
  %i = arith.constant 2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.cmpf "ult", %i, %v : f32
  return %c : i1
}

func.func @uge() -> i1 {
  %i = arith.constant 3.0 : f32
  %v = arith.constant 2.0 : f32
  %c = arith.cmpf "uge", %i, %v : f32
  return %c : i1
}

func.func @ugt() -> i1 {
  %i = arith.constant 3.0 : f32
  %v = arith.constant 2.0 : f32
  %c = arith.cmpf "ugt", %i, %v : f32
  return %c : i1
}
