func.func @int(%x: i32) -> i32 {
  %y = math.absi %x: i32
  return %y: i32
}

func.func @fp(%x: f32) -> f32 {
  %y = math.absf %x: f32
  return %y: f32
}
