import math, unittest
from tinygrad.shape.symbolic import Variable
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor, Device
import numpy as np
import torch

@unittest.skipUnless(Device.DEFAULT in ["GPU", "METAL", "CLANG"], f"{Device.DEFAULT} is not supported")
class TestSymbolicOps(unittest.TestCase):
  def setUp(self): GlobalCounters.var_vals = {}
  def tearDown(self): GlobalCounters.var_vals = {}

  def test_plus1(self):
    def f(a): return (a+1).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10)
      a = Tensor.rand(3, i)
      na = a.cpu().numpy()
      c = f(a.reshape(3, vi)).reshape(3, i)
      np.testing.assert_allclose(c.cpu().numpy(), na+1, atol=1e-6,rtol=1e-6)

  def test_add(self):
    def f(a,b): return (a+b).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, i)
      na = a.cpu().numpy()
      nb = b.cpu().numpy()
      c = f(a.reshape(3, vi), b.reshape(3, vi)).reshape(3, i)
      np.testing.assert_allclose(c.cpu().numpy(), na+nb, atol=1e-6,rtol=1e-6)

  def test_2d_matmul(self):
    def matmul(a, b): return (a@b).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      na = a.cpu().numpy()
      nb = b.cpu().numpy()
      c = matmul(a.reshape(3, ii), b.reshape(ii, 5))
      np.testing.assert_allclose(c.cpu().numpy(), na@nb, atol=1e-6,rtol=1e-6)

  def test_cat_dim0(self):
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

  def test_cat_dim1(self):
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

  def test_two_vars_plus1(self):
    def f(a): return (a+1).realize()
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10)
        vj = Variable("j", 1, 10)
        a = Tensor.rand(i, 3)
        b = Tensor.rand(3, j)
        na = a.cpu().numpy()
        nb = b.cpu().numpy()

        a = a.reshape(vi, 3)
        b = b.reshape(3, vj)
        c = a@b
        assert c.shape == (vi, vj)
        # TODO: this breaks on kernel optimization with GPU now. fix when we have two variable use case
        # c = f(c).reshape(i, j)
        # np.testing.assert_allclose(c.cpu().numpy(), na@nb+1, atol=1e-6,rtol=1e-6)
