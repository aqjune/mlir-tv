#include "tensor.h"
#include "vcgen.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "z3++.h"
#include <functional>
#include <map>
#include <sstream>
#include <variant>
#include <vector>

using namespace mlir;

struct RegFile {
  std::vector<std::pair<mlir::Value, Tensor>> m;

  void add(mlir::Value v, Tensor &&t) {
    m.emplace_back(v, std::move(t));
  }

  Tensor &get(mlir::Value v) {
    for (auto &[a, b]: m)
      if (a == v)
        return b;
    llvm_unreachable("Unknown key");
  }
};

struct State {
  RegFile regs;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, State &);
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, State &s) {
  for (auto itm: s.regs.m) {
    os << "Register: " << itm.first;
    os << "Value: " << itm.second << "\n";
  }
  return os;
}


#define RET_STR(V) { \
  std::string msg; \
  llvm::raw_string_ostream rso(msg); \
  rso << V; \
  rso.flush(); \
  return msg; \
}

static std::variant<std::string, State>
createInputState(FuncOp fn) {
  State s;
  unsigned n = fn.getNumArguments();
  for (unsigned i = 0; i < n; ++i) {
    auto arg = fn.getArgument(i);
    if (auto ty = arg.getType().dyn_cast<mlir::TensorType>()) {
      s.regs.add(arg, Tensor::newVar(ty, std::to_string(arg.getArgNumber())));
    } else {
      RET_STR("Unsupported type: " << arg.getType());
    }
  }

  return s;
}

static std::optional<std::string>
encodeConv(State &st, linalg::ConvInputNHWCFilterHWCFOp op) {
  if (!llvm::all_of(op.dilations(), [](auto i) { return i == 1; }))
    return "dilation isn't one\n";
  else if (!llvm::all_of(op.strides(), [](auto i) { return i == 1; }))
    return "strides isn't one\n";

  auto inputs = op.getInputTensors();
  if (inputs.size() != 2)
    return "conv_2d_input_nhwc_filter_hwcf with (input, filter) input tensors"
           " is supported only";
  auto input = inputs[0];
  auto filter = inputs[1];

  if (op.getNumOutputTensors() != 1)
    return "conv_2d_input_nhwc_filter_hwcf with one output tensor is"
           " supported only";
  auto output = op.getOutputTensors()[0];

  auto &input_z3 = st.regs.get(input);
  auto &filter_z3 = st.regs.get(filter);

  std::vector<z3::expr> output_dims = {
    Tensor::newIdxConst(1),
    input_z3.dims[1] + 1 - filter_z3.dims[0],
    input_z3.dims[2] + 1 - filter_z3.dims[1],
    filter_z3.dims[3]
  };
  std::vector<z3::expr> cube_size = {
    Tensor::newIdxConst(1),
    filter_z3.dims[0], filter_z3.dims[1], filter_z3.dims[2]
  };

  auto vars = Tensor::newIdxVars({"i", "j", "k", "l"});

  // TODO: fill this
  return {};
}

static std::optional<std::string> encode(State &st, FuncOp &fn) {
  for (auto &region: fn) {
    for (auto &op: region) {
      op.dump();
      if (auto nwhcOp = dyn_cast<linalg::ConvInputNHWCFilterHWCFOp>(op)) {
        auto errmsg = encodeConv(st, nwhcOp);
        if (errmsg) {
          RET_STR("Cannot encode " << op << "\n\t" << *errmsg << "\n");
        }
      }
    }
  }
  return {};
}


static void verify(FuncOp src, FuncOp tgt) {
  llvm::outs() << "<" << src.getName() << ">\n";
  assert(src.getNumArguments() == tgt.getNumArguments());

  auto raiseUnsupported = [](const std::string &msg) {
    llvm::errs() << msg << "\n";
    exit(1);
  };

  auto st_src_or_err = createInputState(src);
  if (std::holds_alternative<std::string>(st_src_or_err))
    raiseUnsupported(std::get<std::string>(st_src_or_err));

  auto st_src = std::get<State>(st_src_or_err);
  auto st_tgt = st_src;
  llvm::outs() << st_src << "\n";

  llvm::outs() << "<src>\n";
  if (auto msg = encode(st_src, src))
    raiseUnsupported(*msg);

  llvm::outs() << "\n";
  llvm::outs() << "<tgt>\n";
  if (auto msg = encode(st_tgt, tgt))
    raiseUnsupported(*msg);

  // TODO: compare the final state
}

void verify(OwningModuleRef &src, OwningModuleRef &tgt) {
  std::map<StringRef, FuncOp> srcfns, tgtfns;
  auto fillFns = [](std::map<StringRef, FuncOp> &m, Operation &op) {
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