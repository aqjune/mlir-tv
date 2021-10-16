// VERIFY

// Running –convert-std-to-llvm shows that index_cast is sext

func @index_cast() -> index {
  %c = constant -1: i32
  %y = index_cast %c : i32 to index
  return %y: index
}

