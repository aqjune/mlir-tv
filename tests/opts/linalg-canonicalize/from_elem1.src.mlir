// VERIFY

func.func @from_elem1(%element : index) -> index {
  %c0 = arith.constant 0 : index
  %tensor = tensor.from_elements %element : tensor<index>
  %extracted_element = tensor.extract %tensor[] : tensor<index>
  return %extracted_element : index
}

// How to reproduce tgt:
// mlir-opt -canonicalize <src>