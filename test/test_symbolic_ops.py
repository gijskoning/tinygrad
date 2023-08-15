import unittest, itertools
from tinygrad.shape.symbolic import Variable
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor, Device
import numpy as np
import torch

@unittest.skipIf(getenv("ARM64"), "ARM64 is not supported")
@unittest.skipUnless(Device.DEFAULT in ["GPU", "METAL", "CLANG"], f"{Device.DEFAULT} is not supported")
class TestSymbolicOps(unittest.TestCase):
  def test_plus1(self):
    def f(a): return (a+1).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(3, i)
      symbolic = f(a.reshape(3, vi)).reshape(3, i).cpu().numpy()
      expected = f(a).cpu().numpy()

  def test_add(self):
    def f(a,b): return (a+b).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, i)
      symbolic = f(a.reshape(3, vi), b.reshape(3, vi)).reshape(3, i).cpu().numpy()
      expected = f(a, b).cpu().numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      symbolic = f(a.reshape(3, vi), b.reshape(vi, 5)).cpu().numpy()
      expected = f(a, b).cpu().numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  @unittest.skipIf(Device.DEFAULT == "CLANG", "broken on CLANG CI")
  def test_attention(self):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      q = Tensor.rand(2, 1, 4, 8)
      k = Tensor.rand(2, i, 4, 8)
      v = Tensor.rand(2, i, 4, 8)
      symbolic = f(q, k.reshape(2, vi, 4, 8), v.reshape(2, vi, 4, 8)).reshape(2, 4, 1, 8).cpu().numpy()
      expected = f(q, k, v).cpu().numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim_0(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        a = Tensor.rand(i, 3)
        b = Tensor.rand(j, 3) # j is treated as a constant
        symbolic = f(a.reshape(vi, 3), b).reshape(i+j, 3).cpu().numpy()
        expected = f(a, b).cpu().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim1(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        a = Tensor.rand(3, i)
        b = Tensor.rand(3, j) # j is treated as a constant
        symbolic = f(a.reshape(3, vi), b).reshape(3, i+j).cpu().numpy()
        expected = f(a, b).cpu().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_two_vars_plus1(self):
    def f(a, b): return (a@b+1).realize()
    vi = Variable("i", 1, 10)
    vj = Variable("j", 1, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        a = Tensor.rand(i, 3)
        b = Tensor.rand(3, j)
        symbolic = f(a.reshape(vi, 3), b.reshape(3, vj)).reshape(i, j).cpu().numpy()
        expected = f(a, b).cpu().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)