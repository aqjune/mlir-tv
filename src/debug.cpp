#include "debug.h"
using namespace std;

static bool is_verbose = false;
static string dummy_str;
static llvm::raw_string_ostream dummy_ss(dummy_str);

void setVerbose(bool vb) {
  is_verbose = vb;
}

llvm::raw_ostream &verbose(const string &prefix) {
  dummy_ss.flush();
  dummy_str.clear();
  llvm::raw_ostream *os;
  if (!is_verbose)
    os = &dummy_ss;
  else
    os = &llvm::outs();
  *os << "[" << prefix << "]: ";
  return *os;
}