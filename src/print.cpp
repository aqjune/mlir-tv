#include "print.h"

using namespace std;
using namespace smt;

void printCounterEx(
    model m, const vector<expr> &params, mlir::FuncOp src,
    mlir::FuncOp tgt, const State &st_src, const State &st_tgt,
    VerificationStep step) {
  auto or_omit_z3 = [&](const expr &e) -> string {
    string s;
    llvm::raw_string_ostream rso(s);
    rso << e;
    rso.flush();

    if (s.size() > 500)
      return "(omitted)";
    return s;
  };

  llvm::outs() << "<Inputs>\n";

  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto argsrc = src.getArgument(i);
    llvm::outs() << "\targ" << argsrc.getArgNumber() << ": "
                 << eval(st_src.regs.findOrCrash(argsrc), m)
                 << "\n";
  }

  llvm::outs() << "\n<Source's instructions>\n";
  for (auto &op: src.getRegion().front()) {
    llvm::outs() << "\t" << op << "\n";
    if (op.getNumResults() > 0 && st_src.regs.contains(op.getResult(0))) {
      auto value =  st_src.regs.findOrCrash(op.getResult(0));
      llvm::outs() << "\t\tValue: " << eval(move(value), m) << "\n";
    }
  }

  llvm::outs() << "\n<Target's instructions>\n";
  for (auto &op: tgt.getRegion().front()) {
    llvm::outs() << "\t" << op << "\n";
    if (op.getNumResults() > 0 && st_tgt.regs.contains(op.getResult(0))) {
      auto value = st_tgt.regs.findOrCrash(op.getResult(0));
      llvm::outs() << "\t\tValue: " << eval(move(value), m) << "\n";
    }
    auto wb = m.eval(st_tgt.isOpWellDefined(&op));
    if (wb.is_false()) {
      llvm::outs() << "\t\t[This operation has undefined behavior!]\n";
      break;
    }
  }

  if (st_src.retValue && step == VerificationStep::RetValue) {
    if (src.getNumResults() == 1 &&
        src.getType().getResult(0).isa<mlir::TensorType>()) {
      llvm::outs() << "\n<Returned tensor>\n";

      auto t_src = get<Tensor>(*st_src.retValue).eval(m);
      auto t_tgt = get<Tensor>(*st_tgt.retValue).eval(m);

      llvm::outs() << "Dimensions (src): " << t_src.getDims() << '\n';
      llvm::outs() << "Dimensions (tgt): " << t_tgt.getDims() << '\n';

      if (params.size() > 0) {
        // More than size mismatch
        assert(params.size() == 1);
        auto param = m.eval(params[0]);
        auto indices = simplifyList(from1DIdx(param, t_src.getDims()));
        llvm::outs() << "Index: " << indices << '\n';
        llvm::outs() << "Element (src): "
                    << or_omit_z3(t_src.get(indices).simplify())
                    << '\n';
        llvm::outs() << "Element (tgt): "
                    << or_omit_z3(t_tgt.get(indices).simplify())
                    << '\n';
      }

    } else {
      llvm::outs() << "\n<Returned value>\n";

      for (auto &param: params)
        llvm::outs() << "\tIndex: " << m.eval(param) << "\n";
      llvm::outs() << "\tSrc: " << eval(*st_src.retValue, m) << "\n";
      llvm::outs() << "\tSrc: " << eval(*st_tgt.retValue, m) << "\n";
    }
  }

  if (step == VerificationStep::Memory) {
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