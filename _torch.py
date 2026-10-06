import unittest
try:
    import torch  # noqa: F401
except ImportError:                       # pure-Python tests still run
    raise unittest.SkipTest("torch not installed")
