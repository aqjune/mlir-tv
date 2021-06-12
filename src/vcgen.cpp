#include "vcgen.h"
#include "z3++.h"
#include <functional>
#include <map>
#include <sstream>

using namespace mlir;
using namespace std;

static z3::context c;
const unsigned BITS_FLOAT = 4;
const unsigned BITS_INDEX = 32;


class Tensor {
public:
  vector<z3::expr> dims;
  z3::expr arr;

  Tensor(): arr(c) {}

  static Tensor newVar(mlir::TensorType tensorTy, const std::string &name) {
    Tensor t;

    uint64_t rank = tensorTy.getRank();
    for (auto i = 0; i < rank; ++i) {
      t.dims.emplace_back(c.bv_val(tensorTy.getDimSize(i), BITS_INDEX));
    }
    t.arr = c.constant(name.c_str(),
          c.array_sort(c.bv_sort(BITS_INDEX), c.bv_sort(BITS_FLOAT)));

    return t;
  }

  friend ostream& operator<<(ostream&, Tensor &);
};

ostream& operator<<(ostream& os, Tensor &t) {
  os << t.arr << "(dim :" << t.dims[0];
  for (size_t i = 1; i < t.dims.size(); ++i)
    os << ", " << t.dims[i];
  os << ")";
  return os;
};


struct RegFile {
  map<string, Tensor> m;
};

struct State {
  RegFile regs;

  friend ostream& operator<<(ostream&, State &);
};

ostream& operator<<(ostream& os, State &s) {
  for (auto itm: s.regs.m) {
    os << itm.first << ": " << itm.second << "\n";
  }
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &ros, State &s) {
  stringstream ss;
  ss << s;
  ros << ss.str();
  return ros;
};

static State createInputState(FuncOp fn) {
  State s;
  unsigned n = fn.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto arg = fn.getArgument(i);
    if (auto ty = arg.getType().dyn_cast<mlir::TensorType>()) {
      auto name = to_string(arg.getArgNumber());
      s.regs.m.emplace(name, Tensor::newVar(ty, name));
    } else {
      llvm::errs() << "Unsupported type: " << arg.getType() << "\n";
      exit(1);
    }
  }

  return s;
}

static void encode(State &st, FuncOp &fn) {
  for (auto &region: fn) {
    for (auto &op: region) {
      op.dump();
    }
  }
}

static void verify(FuncOp src, FuncOp tgt) {
  llvm::outs() << "<" << src.getName() << ">\n";
  assert(src.getNumArguments() == tgt.getNumArguments());

  auto st_src = createInputState(src);
  auto st_tgt = st_src;
  llvm::outs() << st_src << "\n";

  llvm::outs() << "<src>\n";
  encode(st_src, src);

  llvm::outs() << "\n";
  llvm::outs() << "<tgt>\n";
  encode(st_tgt, tgt);

  // TODO: compare the final state
}

void verify(OwningModuleRef &src, OwningModuleRef &tgt) {
  map<StringRef, FuncOp> srcfns, tgtfns;
  auto fillFns = [](map<StringRef, FuncOp> &m, Operation &op) {
    auto fnop = dyn_cast<FuncOp>(op);
    m[fnop.getName()] = fnop;
  };
  llvm::for_each(*src, [&](auto &op) { fillFns(srcfns, op); });
  llvm::for_each(*tgt, [&](auto &op) { fillFns(tgtfns, op); });

  for (auto [name, srcfn]: srcfns) {
    auto itr = tgtfns.find(name);
    if (itr == tgtfns.end()) {
      // The function does not exist in tgt! Let's skip this.
      // TODO: we should notify users that the functions are not checked.
      continue;
    }
    // TODO: check fn signature
    verify(srcfn, itr->second);
  }
}