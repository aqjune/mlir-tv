// VERIFY

func.func @int(%x: i32) -> i32 {
  %y = math.absi %x: i32
  %z = math.absi %y: i32
  return %z: i32
}

func.func @fp(%x: f32) -> f32 {
  %y = math.absf %x: f32
  %z = math.absf %y: f32
  return %z: f32
}


