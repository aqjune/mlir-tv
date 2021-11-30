#include "utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace std;

string to_string(mlir::Type t) {
  string ss;
  llvm::raw_string_ostream os(ss);
  os << t;
  os.flush();
  return ss;
}
