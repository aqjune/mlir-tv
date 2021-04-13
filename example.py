from z3 import *
import time

DEBUG=True
BITS_INDEX = 32
BITS_FLOAT = 4


tmpidx = 0
def nextTmpId() -> str:
  global tmpidx
  tmpidx = tmpidx + 1
  return tmpidx

def toBitVecs(idxs_const, bw):
  return [BitVecVal(i, bw) for i in idxs_const]

def BVToPyInt(bv):
  if is_bv_value(bv):
    return bv.as_signed_long()
  return None

def isBVInt(bv, i):
  return is_bv_value(bv) and bv.as_signed_long() == i

def to1DIdx(idxs, sizes):
  assert(len(idxs) == len(sizes))
  idx = idxs[0]
  for i in range(1, len(sizes)):
    if not isBVInt(sizes[i], 1):
      idx = idx * sizes[i]
    idx = idx + idxs[i]
  return idx

def fitsInSize(idxs, sizes):
  preconds = []
  for i in range(0, len(idxs)):
    preconds.append(ULT(idxs[i], sizes[i]))
  return And(preconds)

def ForAllInRanges(idxs, sizes, body):
  assert(len(idxs) == len(sizes))
  idxs = list(idxs)
  sizes = list(sizes)
  i = 0
  while i < len(idxs):
    if isBVInt(sizes[i], 1):
      body = substitute(body, (idxs[i], BitVecVal(0, BITS_INDEX)))
      del sizes[i]
      del idxs[i]
    else:
      i = i + 1

  return ForAll(idxs, Implies(fitsInSize(idxs, sizes), body))


class MemRef:
  def __init__(self, name: str, dims: int, **kwargs):
    self.name = name
    self.ns = []
    if "ns" in kwargs:
      self.ns = list(kwargs["ns"])
      assert(len(self.ns) == dims)
    else:
      for i in range(0, dims):
        self.ns.append(BitVec("%s_dim%d" % (name, i), BITS_INDEX))
    self.val = Array("%s_val" % name, BitVecSort(BITS_INDEX),
                                      BitVecSort(BITS_FLOAT))

  def get(self, idxs):
    return Select(self.val, to1DIdx(idxs, self.ns))

  def to1DArrayWithOfs(self, idxs, sizes):
    assert(len(idxs) == len(sizes))
    idx0 = BitVec("idx", BITS_INDEX)
    idx  = idx0
    indices = []
    szaccml = BitVecVal(1, BITS_INDEX)
    for i in range(0, len(sizes)):
      sz = sizes[len(sizes) - i - 1]
      if isBVInt(sz, 1) or (isinstance(sz, int) and sz == 1):
        indices.append(BitVecVal(0, BITS_INDEX))
        continue

      szaccml = simplify(szaccml * sz)
      indices.append(URem(idx, sz) + idxs[i])
      idx = UDiv(idx, sz)

    return Lambda([idx0], If(ULT(idx0, szaccml), self.get(indices), 0))

  def reshape(self, ns2):
    # Supported only when self.ns is constant
    newmr = MemRef("%s_reshape%d" % (self.name, nextTmpId()),
                   len(ns2), ns=ns2)
    for i in range(0, len(ns2)):
      newmr.ns[i] = simplify(newmr.ns[i])
    newmr.val = self.val
    return newmr


def dot(a, b, n):
  # a, b: Array
  bis = BitVecSort(BITS_INDEX)
  bfs = BitVecSort(BITS_FLOAT)
  ars = ArraySort(bis, bfs)
  i = BitVec("idx", bis)
  dotfn = Function("dot", ars, ars, bfs)
  return dotfn(
    Lambda([i], If(ULT(i, n), Select(a, i), 0)),
    Lambda([i], If(ULT(i, n), Select(b, i), 0)))


def convolution(inp: MemRef, filtr: MemRef, preconds):
  # TODO: expand this to an input with batch size > 1
  assert(len(inp.ns) == 4 and len(filtr.ns) == 4 and isBVInt(inp.ns[0], 1))

  # 1. Make a filter that has dimension f as its primary index
  filter_ffirst = MemRef(filtr.name + "_tmp%d" % nextTmpId(), 4, ns=filtr.ns)
  dim_f = filter_ffirst.ns.pop()
  filter_ffirst.ns.insert(0, dim_f)

  i, j, k, l = BitVecs("i j k l", BITS_INDEX) # h, w, c, f
  idxs = [i, j, k, l]
  preconds.append(
    ForAllInRanges(idxs, filtr.ns,
                   filtr.get([i, j, k, l]) == filter_ffirst.get([l, i, j, k])))

  # 2. evaluate the result using dot
  output_ns = [
      1,           # TODO: support an input with batch size > 1
      inp.ns[1] + 1 - filtr.ns[0],
      inp.ns[2] + 1 - filtr.ns[1],
      filtr.ns[3]] # channel(inp.ns[3] = filtr.ns[2]) disappears
  output = MemRef("conv_output%d" % nextTmpId(), 4, ns=output_ns)
  h_half = LShR(filtr.ns[0], 1)
  w_half = LShR(filtr.ns[1], 1)

  cube_size = [1] + filtr.ns[0:3]
  input_cube = inp.to1DArrayWithOfs(
      [0, i, j, 0], # batch: 0, img size: (h, w), channel starts from zero
      cube_size)

  # get l'th filter
  filter_cube = filter_ffirst.to1DArrayWithOfs([l, 0, 0, 0], cube_size)

  res = dot(input_cube, filter_cube,
            cube_size[0] * cube_size[1] * cube_size[2] * cube_size[3])
  idxs = [k, i, j, l]
  preconds.append(ForAllInRanges(idxs, output.ns, output.get(idxs) == res))

  if DEBUG:
    print("convolution(): result memref size: %s" % str(output.ns))

  return output


def convertImageToMatrix(inp: MemRef, filtr_ns, preconds):
  mat_ns = [
    inp.ns[0],
    inp.ns[1] - filtr_ns[0] + 1,
    inp.ns[2] - filtr_ns[1] + 1,
    filtr_ns[0],
    filtr_ns[1],
    filtr_ns[2]
  ]
  mat_ns = [simplify(x) for x in mat_ns]
  mat = MemRef("imgtomat_%s%d" % (inp.name, nextTmpId()), 6, ns=mat_ns)

  i, j, k, l, m, n = BitVecs("i j k l m n", BITS_INDEX)
  idxs = [i, j, k, l, m, n]
  si, sj, sk, sl, sm, sn = (
    inp.ns[0], # batch
    inp.ns[1] - filtr_ns[0] + 1, # h
    inp.ns[2] - filtr_ns[1] + 1, # w
    filtr_ns[0], # fh
    filtr_ns[1], # fw
    filtr_ns[2])
  sizes = [si, sj, sk, sl, sm, sn]

  res = mat.get([i, j, k, l, m, n])
  preconds.append(
      ForAllInRanges(idxs, sizes, inp.get([i, j + l, k + m, n]) == res))

  if DEBUG:
    print("convertImageToMatrix(): result memref size: %s" % str(mat.ns))

  return mat


def matmul(a: MemRef, b: MemRef, preconds):
  assert(len(a.ns) == 2 and len(b.ns) == 2)
  bt = MemRef("%s_transpose%d" % (b.name, nextTmpId()), 2,
              ns=[b.ns[1], b.ns[0]])

  i, j = BitVecs("i j", BITS_INDEX)
  preconds.append(ForAllInRanges([i, j], b.ns, b.get([i, j]) == bt.get([j, i])))

  output = MemRef("matmul%d" % nextTmpId(), 2, ns=[a.ns[0], bt.ns[0]])

  a_row = a.to1DArrayWithOfs([i, 0], [1, a.ns[1]])
  bt_row = bt.to1DArrayWithOfs([j, 0], [1, bt.ns[1]])

  preconds.append(
      ForAllInRanges([i, j], output.ns,
                     output.get([i, j]) == dot(a_row, bt_row, a.ns[1])))

  if DEBUG:
    print("matmul(): result memref size: %s" % str(output.ns))

  return output




# Inputs
image = MemRef("image",  4, ns=toBitVecs([1, 4, 4, 1], BITS_INDEX))
filtr = MemRef("filter", 4, ns=toBitVecs([3, 3, 1, 1], BITS_INDEX))
#image = MemRef("image",  4, ns=toBitVecs([1, 16, 16, 4], BITS_INDEX))
#filtr = MemRef("filter", 4, ns=toBitVecs([3, 3, 4, 16], BITS_INDEX))

# Preconditions
preconds = []

# Source program
output_src = convolution(image, filtr, preconds)

# Target program
mat = convertImageToMatrix(image, filtr.ns, preconds)

mat = mat.reshape([
    image.ns[0] * (image.ns[1] - filtr.ns[0] + 1)
                * (image.ns[2] - filtr.ns[1] + 1),
    filtr.ns[0] * filtr.ns[1] * filtr.ns[2]
])
filtr = filtr.reshape([filtr.ns[0] * filtr.ns[1] * filtr.ns[2],
                       filtr.ns[3]])

output_tgt = matmul(mat, filtr, preconds)


s = SolverFor("UFBV")

# Goal
s.add(And(preconds))

i_counterex = BitVec("i_counterex", BITS_INDEX)
neg_goal = And(ULT(i_counterex, output_tgt.ns[0] * output_tgt.ns[1]),
                Select(output_src.val, i_counterex) !=
                Select(output_tgt.val, i_counterex))
s.add(neg_goal)


with open("dump.txt", mode='w') as f:
  f.write("\n".join([str(x) for x in preconds]))
  f.write("\n" + str(neg_goal))
with open("dump.smt2", mode='w') as f:
  f.write(s.to_smt2())

# Solve
timeStart = time.time()
result = s.check()
timeEnd = time.time()

def z3ResToStr(result):
  if result == unsat:
    return "CORRECT"
  elif result == sat:
    return "INCORRECT"
  return "UNKNOWN"

print("== Result: %s ==\nRunning time: %s secs" % 
      (z3ResToStr(result), timeEnd - timeStart))
if result == unknown:
  print(s.reason_unknown())
elif result == sat:
  model = s.model()
  print(model)
