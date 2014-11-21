import os
import unittest

import testchill.omega
import testchill.util


class TestOmegaTestCases(unittest.TestCase):
    def setUp(self):
        self.omega_dev_dir = os.getenv('OMEGA_DEV_SRC')
        self.omega_rel_dir = os.getenv('OMEGA_RELEASE_SRC')
    
    def tearDown(self):
        pass
    
    def test_omega_dev(self):
        tc = testchill.omega.BuildOmegaTestCase(self.omega_dev_dir)
        tc.run()
        
    def test_omega_release(self):
        tc = testchill.omega.BuildOmegaTestCase(self.omega_rel_dir, 'release')
        tc.run()
    
