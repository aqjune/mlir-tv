#include "abstractops.h"
#include "print.h"

using namespace std;
using namespace smt;

static void printInputs(Model m, mlir::FuncOp src, const State &st_src) {
  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto argsrc = src.getArgument(i);
    llvm::outs() << "\targ" << argsrc.getArgNumber() << ": "
                 << eval(st_src.regs.findOrCrash(argsrc), m)
                 << "\n";
  }
}

void printOperations(Model m, mlir::FuncOp fn, const State &st) {
  for (auto &op: fn.getRegion().front()) {
    llvm::outs() << "\t" << op << "\n";

    auto wb = m.eval(st.isOpWellDefined(&op));
    if (wb.isFalse()) {
      llvm::outs() << "\t\t[This operation has undefined behavior!]\n";
      break;
    }

    if (op.getNumResults() > 0 && st.regs.contains(op.getResult(0))) {
      auto value = st.regs.findOrCrash(op.getResult(0));
      llvm::outs() << "\t\tValue: " << eval(move(value), m) << "\n";
    }
  }
}

void printCounterEx(
    Model m, const vector<Expr> &params, mlir::FuncOp src,
    mlir::FuncOp tgt, const State &st_src, const State &st_tgt,
    VerificationStep step, unsigned retvalidx) {
  llvm::outs() << "<Inputs>\n";
  printInputs(m, src, st_src);

  llvm::outs() << "\n<Source's instructions>\n";
  printOperations(m, src, st_src);

  llvm::outs() << "\n<Target's instructions>\n";
  printOperations(m, tgt, st_tgt);


  if (step == VerificationStep::RetValue) {
    if (src.getType().getResult(retvalidx).isa<mlir::TensorType>()) {
      llvm::outs() << "\n<Returned tensor>\n";

      auto t_src = get<Tensor>(st_src.retValues[retvalidx]).eval(m);
      auto t_tgt = get<Tensor>(st_tgt.retValues[retvalidx]).eval(m);

      llvm::outs() << "Dimensions (src): " << or_omit(t_src.getDims()) << '\n';
      llvm::outs() << "Dimensions (tgt): " << or_omit(t_tgt.getDims()) << '\n';

      if (params.size() > 0) {
        // More than size mismatch
        assert(params.size() == 1);
        auto param = m.eval(params[0]);
        auto indices = simplifyList(from1DIdx(param, t_src.getDims()));
        llvm::outs() << "Index: " << or_omit(indices) << '\n';
        llvm::outs() << "Element (src): "
                    << or_omit(t_src.get(indices).first.simplify())
                    << '\n';
        llvm::outs() << "Element (tgt): "
                    << or_omit(t_tgt.get(indices).first.simplify())
                    << '\n';
      }

    } else {
      llvm::outs() << "\n<Returned value>\n";

      for (auto &param: params)
        llvm::outs() << "\tIndex: " << m.eval(param) << "\n";

      llvm::outs() << "\tSrc: " << eval(st_src.retValues[retvalidx], m)
                   << "\n";
      llvm::outs() << "\tTgt: " << eval(st_tgt.retValues[retvalidx], m)
                   << "\n";
    }
  } else if (step == VerificationStep::Memory) {
    // Print Memory counter example
    auto bid = params[0];
    auto offset = params[1];

    auto [srcValue, srcSuccess] = st_src.m->load(bid, offset);
    auto [tgtValue, tgtSuccess] = st_tgt.m->load(bid, offset);
    auto srcWritable = st_src.m->getWritable(bid);
    auto tgtWritable = st_tgt.m->getWritable(bid);

    srcValue = m.eval(srcValue, true);
    srcSuccess = m.eval(srcSuccess);
    tgtValue = m.eval(tgtValue, true);
    tgtSuccess = m.eval(tgtSuccess);
    srcWritable = m.eval(srcWritable);
    tgtWritable = m.eval(tgtWritable);

    llvm::outs() << "\n<Source memory state>\n";
    llvm::outs() << "\tMemory[bid: " << m.eval(bid)
      << ", offset: " << m.eval(offset) << "] : "
      << srcValue << ", " << srcWritable <<  "\n";
    llvm::outs() << "\n<Target memory state>\n";
    llvm::outs() << "\tMemory[bid: " << m.eval(bid)
      << ", offset: " << m.eval(offset) << "] : "
      << tgtValue << ", " << tgtWritable <<  "\n\n";
  }
}
