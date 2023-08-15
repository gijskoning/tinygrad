import math, unittest
from tinygrad.jit import TinyJit
from tinygrad.helpers import GlobalCounters
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor, Device
import numpy as np
import torch

@unittest.skipUnless(Device.DEFAULT in ["GPU", "METAL", "CLANG"], f"{Device.DEFAULT} is not supported")
class TestSymbolicJit(unittest.TestCase):
  def setUp(self): GlobalCounters.var_vals = {}
  def tearDown(self) -> None: GlobalCounters.var_vals = {}

  def test_2d_matmul(self):
    @TinyJit
    def matmul(a, b): return (a@b).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      na = a.cpu().numpy()
      nb = b.cpu().numpy()
      c = matmul(a.reshape(3, ii), b.reshape(ii, 5))
      np.testing.assert_allclose(c.cpu().numpy(), na@nb, atol=1e-6,rtol=1e-6)
    assert len(matmul.jit_cache) == 1

  def test_mixed_with_no_symbol_kernel(self):
    @TinyJit
    def matmul(a, b):
      s = (a@b).realize()
      s = (s+s).realize() # this one does not have symbols in input
      return s
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      na = a.cpu().numpy()
      nb = b.cpu().numpy()
      s = na@nb
      s = s+s
      c = matmul(a.reshape(3, ii), b.reshape(ii, 5))
      np.testing.assert_allclose(c.cpu().numpy(), s, atol=1e-6, rtol=1e-6)
    assert len(matmul.jit_cache) == 2

  def test_cat_dim0(self):
    @TinyJit
    def cat(a, b): return a.cat(b, dim=0).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(i, 3)
      b = Tensor.rand(1, 3)
      na = a.cpu().numpy()
      nb = b.cpu().numpy()
      s = np.concatenate([na,nb],axis=0)
      c = cat(a.reshape(ii, 3), b).reshape(i+1, 3)
      np.testing.assert_equal(c.cpu().numpy(), s)
    assert len(cat.jit_cache) == 1

  def test_cat_dim1(self):
    @TinyJit
    def cat(a, b): return a.cat(b, dim=1).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, 1)
      na = a.cpu().numpy()
      nb = b.cpu().numpy()
      s = np.concatenate([na,nb],axis=1)
      c = cat(a.reshape(3, ii), b).reshape(3, i+1)
      np.testing.assert_equal(c.cpu().numpy(), s)
    assert len(cat.jit_cache) == 1
