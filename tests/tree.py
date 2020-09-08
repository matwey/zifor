from zifor.tree import Tree
import numpy as np
import numpy.ma as ma
import unittest

class TreeTest(unittest.TestCase):
	def setUp(self):
		pass

	def test_create1(self):
		rng = np.random.default_rng(42)
		t = Tree(5, 100, rng)

	def test_fit1(self):
		rng = np.random.default_rng(42)
		t = Tree(10, 100, rng)
		x = ma.array(np.asarray(1.0 / np.linspace(1.0, 20.0, 10)), mask=np.array([False,]*10))
		x = x.reshape(-1,1)
		t.fit(x)

	def test_predict1(self):
		rng = np.random.default_rng(42)
		t = Tree(10, 100, rng)
		x = ma.array(np.asarray(1.0 / np.linspace(1.0, 20.0, 10)), mask=np.array([False,]*10))
		x = x.reshape(-1,1)
		t.fit(x)
		depth = t.predict(x)
		np.testing.assert_allclose(depth.flatten(), np.array([2,3,4,6,7,8,9,9,6,6], dtype=np.float32))
