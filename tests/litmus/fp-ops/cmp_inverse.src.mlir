// VERIFY

func.func @ole(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ole", %arg0, %arg1 : f32
  return %c : i1
}

func.func @olt(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "olt", %arg0, %arg1 : f32
  return %c : i1
}

func.func @oge(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "oge", %arg0, %arg1 : f32
  return %c : i1
}

func.func @ogt(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ogt", %arg0, %arg1 : f32
  return %c : i1
}

func.func @ule(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ule", %arg0, %arg1 : f32
  return %c : i1
}

func.func @ult(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ult", %arg0, %arg1 : f32
  return %c : i1
}

func.func @uge(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "uge", %arg0, %arg1 : f32
  return %c : i1
}

func.func @ugt(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ugt", %arg0, %arg1 : f32
  return %c : i1
}
