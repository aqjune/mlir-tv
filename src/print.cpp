#include "abstractops.h"
#include "print.h"

using namespace std;
using namespace smt;

static string intToStr(Expr e) {
  uint64_t u;
  if (e.isUInt(u)) {
    return to_string(u);
  } else {
    stringstream ss;
    ss << e;
    return ss.str();
  }
}

static void printInputs(Model m, mlir::FuncOp src, const State &st_src) {
  unsigned n = src.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto argsrc = src.getArgument(i);
    llvm::outs() << "\targ" << argsrc.getArgNumber() << " ("
        << argsrc.getType () << "): "
        << eval(st_src.regs.findOrCrash(argsrc), m) << "\n";
  }

  llvm::outs() << "  Input memory:\n";
  auto &mem = *st_src.m;
  auto btys = mem.getBlockTypes();
  for (auto &bty: btys) {
    llvm::outs() << "\tType " << bty << ":\n";
    unsigned num = mem.getNumBlocks(bty);

    for (unsigned i = 0; i < num; ++i) {
      auto numelem = m.eval(mem.getNumElementsOfMemBlock(bty, mem.mkBID(i)));
      auto liveness = m.eval(mem.getLiveness(bty, mem.mkBID(i)));
      llvm::outs() << "\t  Block " << i << ": # elems: "
          << intToStr(m.eval(numelem))
          << "\n";
    }
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
    VerificationStep step, unsigned retvalidx, optional<mlir::Type> memElemTy) {
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
      auto elemTy = t_src.getElemType();
      assert(elemTy == t_tgt.getElemType());

      llvm::outs() << "Dimensions (src): " << or_omit(t_src.getDims()) << '\n';
      llvm::outs() << "Dimensions (tgt): " << or_omit(t_tgt.getDims()) << '\n';

      if (params.size() > 0) {
        // More than size mismatch
        assert(params.size() == 1);
        auto param = m.eval(params[0]);
        auto indices = simplifyList(from1DIdx(param, t_src.getDims()));
        llvm::outs() << "Index: " << or_omit(indices) << '\n';

        auto srcElem = fromExpr(t_src.get(indices).simplify(), elemTy);
        auto tgtElem = fromExpr(t_tgt.get(indices).simplify(), elemTy);
        llvm::outs() << "Element (src): " << *srcElem << '\n';
        llvm::outs() << "Element (tgt): " << *tgtElem << '\n';
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
    auto elemTy = *memElemTy;

    auto [srcValue, srcInfo] = st_src.m->load(elemTy, bid, offset);
    auto [tgtValue, tgtInfo] = st_tgt.m->load(elemTy, bid, offset);
    auto srcWritable = st_src.m->getWritable(elemTy, bid);
    auto srcNumElems = st_src.m->getNumElementsOfMemBlock(elemTy, bid);
    auto srcLiveness = st_src.m->getLiveness(elemTy, bid);
    auto tgtWritable = st_tgt.m->getWritable(elemTy, bid);
    auto tgtLiveness = st_tgt.m->getLiveness(elemTy, bid);

    bid = m.eval(bid);
    optional<unsigned> bid_int = bid.asUInt();
    offset = m.eval(offset);
    srcValue = m.eval(srcValue, true);
    tgtValue = m.eval(tgtValue, true);
    srcWritable = m.eval(srcWritable);
    tgtWritable = m.eval(tgtWritable);
    srcNumElems = m.eval(srcNumElems);
    srcLiveness = m.eval(srcLiveness);
    tgtLiveness = m.eval(tgtLiveness);

    llvm::outs() << "\n<Final state of the mismatched memory>\n";
    llvm::outs() << "\tBlock id: " << intToStr(bid);
    if (bid_int) {
      if (auto glbname = st_src.m->getGlobalVarName(elemTy, *bid_int))
        llvm::outs() << " (\"" << *glbname << "\")";
    }
    llvm::outs() << "\n";
    llvm::outs() << "\t\telement type: " << to_string(elemTy) << "\n";
    llvm::outs() << "\t\t# elements: " << intToStr(srcNumElems) << "\n";
    llvm::outs() << "\t\tis writable (src): " << srcWritable << "\n";
    llvm::outs() << "\t\tis writable (tgt): " << tgtWritable << "\n";
    llvm::outs() << "\t\tliveness (src): " << srcLiveness << "\n";
    llvm::outs() << "\t\tliveness (tgt): " << tgtLiveness << "\n";
    llvm::outs() << "\tMismatched element offset: " << intToStr(offset) << "\n";
    llvm::outs() << "\tSource value: " << srcValue << "\n";
    llvm::outs() << "\tTarget value: " << tgtValue << "\n\n";
  }
}
