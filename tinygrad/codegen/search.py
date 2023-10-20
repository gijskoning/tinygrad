from typing import Dict, List, cast, DefaultDict, Optional

from extra.optimization.helpers import lin_to_feats
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Device, Compiled, MemBuffer
from tinygrad.helpers import prod, getenv, flatten,ImageDType, DEBUG
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.runtime.lib import RawBuffer
from collections import defaultdict

from tinygrad.codegen.optimizer import Opt, OptOps
actions = flatten([[Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,7]] for axis in range(6)])
actions += flatten([[Opt(op=OptOps.UNROLL, axis=axis, amt=amt) for amt in [0,4]] for axis in range(4)])
actions += flatten([[Opt(op=OptOps.LOCAL, axis=axis, amt=amt) for amt in [2,3,4,8,16]] for axis in range(5)])
actions += [
  Opt(op=OptOps.LOCAL, axis=0, amt=32),
  Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.GROUP, axis=2, amt=8),
  Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.GROUPTOP, axis=1, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.GROUPTOP, axis=2, amt=256)
]

# returns time in seconds
import shelve
logtm = shelve.open(getenv("LOGTM", "")) if getenv("LOGTM", "") else None
def time_linearizer(lin:Linearizer, rawbufs:List[RawBuffer], allow_test_size=True, max_global_size=65536, cnt=3, should_copy=True) -> float:
  key = str((lin.ast, lin.applied_opts))
  if should_copy and logtm is not None and key in logtm: return min(logtm[key])  # pylint: disable=E1135 # NOTE: we check should_copy since this may have side effects
  if should_copy: lin = lin.copy() # TODO: remove the need for this
  var_vals = {k:k.min for k in vars_from_ast(lin.ast)}
  try:
    lin.linearize()
    prg = cast(Compiled, Device[Device.DEFAULT]).to_program(lin)
    real_global_size = prg.global_size[:]
    if allow_test_size:
      test_global_size = prg.global_size[:]
      while prod(test_global_size) > max_global_size:
        for j in range(2,-1,-1):
          if test_global_size[j] > 16:
            test_global_size[j] //= 2
            break
      factor = prod(prg.global_size) / prod(test_global_size)
      prg.global_size = test_global_size
      #print(real_global_size, test_global_size, factor)
    else:
      factor = 1
    # TODO: this is super broken for var_vals
    global_size, local_size = prg.launch_dims(var_vals)
    tms = [prg.clprg(global_size, local_size, *rawbufs, *var_vals.values(), wait=True)*factor for _ in range(cnt)]
    prg.global_size = real_global_size
  except Exception as e:
    print("FAILED", e)
    #print(lin.ast)
    #print(lin.applied_opts)
    tms = [float('inf')]
  if logtm is not None: logtm[key] = tms
  return min(tms)

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Linearizer) -> List[RawBuffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[RawBuffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    rawbufs[k] = cast(Compiled, Device[Device.DEFAULT]).buffer(prod(lx[0].dtype.shape) if isinstance(lx[0].dtype, ImageDType) else max(y.st.size() for y in lx), lx[0].dtype)
  assert all(r is not None for r in rawbufs)
  return cast(List[RawBuffer], rawbufs)

# get dictionary of all possible actions
def get_linearizer_actions(lin:Linearizer, include_0=True) -> Dict[int, Linearizer]:
  acted_lins = {0:lin.copy()} if include_0 else {}
  for i,a in enumerate(actions):
    if a.axis >= lin.shape_len: continue
    if lin.full_shape[a.axis] == a.amt and Opt(a.op, a.axis, 0) in actions: continue
    lin2 = lin.copy()
    try:
      lin2.apply_opt(a)
      try:
        lin_to_feats(lin2)
      except IndexError as e:
        pass
      up, lcl = 1, 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        if c in {"cyan", "green", "white"}: lcl *= s
      if up > 256 or lcl > 256: continue
      acted_lins[i+1] = lin2
    except Exception:
      pass
  return acted_lins

def beam_search(lin, rawbufs, amt):
  best_tm = float('inf')
  beam: List[Linearizer] = [lin]
  while 1:
    acted_lins = flatten([get_linearizer_actions(lin, include_0=False).values() for lin in beam])
    timed_lins = [(v,time_linearizer(v, rawbufs)) for v in acted_lins]
    opts = sorted(timed_lins, key=lambda x: x[1])
    if len(opts) == 0 or best_tm <= opts[0][1]: break  # we didn't get faster
    best_tm = opts[0][1]
    beam = [x[0] for x in opts[:amt]]
    if DEBUG >= 1: print(f"{opts[0][1]*1e3:10.2f} ms from {len(opts):3d} actions", beam[0].colored_shape())
  return beam[0]