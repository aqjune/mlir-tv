func @ole(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "oge", %arg1, %arg0 : f32
  return %c : i1
}

func @olt(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ogt", %arg1, %arg0 : f32
  return %c : i1
}

func @oge(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ole", %arg1, %arg0 : f32
  return %c : i1
}

func @ogt(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "olt", %arg1, %arg0 : f32
  return %c : i1
}

func @ule(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "uge", %arg1, %arg0 : f32
  return %c : i1
}

func @ult(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ugt", %arg1, %arg0 : f32
  return %c : i1
}

func @uge(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ule", %arg1, %arg0 : f32
  return %c : i1
}

func @ugt(%arg0: f32, %arg1: f32) -> i1 {
  %c = arith.cmpf "ult", %arg1, %arg0 : f32
  return %c : i1
}
