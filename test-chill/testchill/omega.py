from . import test
from . import util



class BuildOmegaTestCase(test.TestCase):
    def __init__(self, omega_dir, version='dev'):
        super(BuildOmegaTestCase, self).__init__(BuildOmegaTestCase.getname(version))
        self.omega_dir = omega_dir
        self.version = version
    
    @staticmethod
    def getname(version):
        if version == 'release':
            return 'omega-release'
        else:
            return 'omega'
    
    def setUp(self):
        util.shell('make clean', wd=self.omega_dir)
    
    def tearDown(self):
        pass
    
    def run(self):
        util.shell('make depend', wd=self.omega_dir)
        util.shell('make', wd=self.omega_dir)


