#include "vcgen.h"
#include <functional>
#include <map>

using namespace mlir;
using namespace std;

struct RegFile {
  // TODO: the value type should be z3 expr
  map<string, int> m;
};

struct State {
  RegFile regs;
};

static State createInputState(FuncOp fn) {
  State s;
  unsigned n = fn.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto arg = fn.getArgument(i);
    // FIXME
    s.regs.m[to_string(arg.getArgNumber())] = 0;
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