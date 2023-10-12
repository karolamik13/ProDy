"""This module contains unit tests for :mod:`~prody.proteins`."""

import os

import numpy as np
from numpy.testing import *
try:
    import numpy.testing.decorators as dec
except ImportError:
    from numpy.testing import dec

from prody import *
from prody import LOGGER
from prody.utilities import which
from prody.tests import TEMPDIR, unittest
from prody.tests.datafiles import *

LOGGER.verbosity = 'none'

class TestParseMMCIF(unittest.TestCase):

    def setUp(self):
        """Set MMCIF file data and parse the MMCIF file."""
        self.multi = DATA_FILES['multi_model_cif']
        self.no_pdb = DATA_FILES['long_chid_cif']
        self.biomols = DATA_FILES['biomols_cif']

    def testUsualCase(self):
        """Test the outcome of a simple parsing scenario."""

        ag = parseDatafile(self.multi['file'])

        self.assertIsInstance(ag, prody.AtomGroup,
            'parseMMCIF failed to return an AtomGroup instance')

        self.assertEqual(ag.numAtoms(), self.multi['atoms'],
            'parseMMCIF failed to parse correct number of atoms')

        self.assertEqual(ag.numCoordsets(), self.multi['models'],
            'parseMMCIF failed to parse correct number of coordinate sets '
            '(models)')

        self.assertEqual(ag.getTitle(),
             os.path.splitext(self.multi['file'])[0],
            'failed to set AtomGroup title based on filename')

    def testPDBArgument(self):
        """Test outcome of invalid *pdb* arguments."""

        self.assertRaises(IOError, parseMMCIF, self.multi['file'] + '.gz')
        self.assertRaises(TypeError, parseMMCIF, None)

    def testModelArgument(self):
        """Test outcome of valid and invalid *model* arguments."""

        path = pathDatafile(self.multi['file'])
        self.assertRaises(TypeError, parseMMCIF, path, model='0')
        self.assertRaises(ValueError, parseMMCIF, path, model=-1)
        self.assertRaises(proteins.MMCIFParseError, parseMMCIF, path,
                          model=self.multi['models']+1)
        self.assertIsNone(parseMMCIF(path, model=0),
            'parseMMCIF failed to parse no coordinate sets')

        self.assertEqual(parseMMCIF(path, model=1).numCoordsets(), 1,
            'parseMMCIF failed to parse the first coordinate set')

        self.assertEqual(parseMMCIF(path, model=2).numCoordsets(), 1,
            'parseMMCIF failed to parse the 2nd coordinate set')

        self.assertEqual(parseMMCIF(path, model=1).numAtoms(), 
                        self.multi['atoms'],
                        'parseMMCIF failed to parse the 1st coordinate set')

        self.assertEqual(parseMMCIF(path, model=2).numAtoms(), 
                        self.multi['atoms'],
                        'parseMMCIF failed to parse the 2nd coordinate set')
            
        self.assertEqual(parseMMCIF(path, 
                                    model=self.multi['models']).numCoordsets(), 
                        1, 'parseMMCIF failed to parse the last coordinate set')

    def testTitleArgument(self):
        """Test outcome of *title* argument."""

        path = pathDatafile(self.multi['file'])
        title = 'small protein'
        self.assertEqual(parseMMCIF(path, title=title).getTitle(),
             title, 'parseMMCIF failed to set user given title')

        name = 1999
        self.assertEqual(parseMMCIF(path, title=name).getTitle(),
             str(name), 'parseMMCIF failed to set user given non-string name')

    def testChainArgument(self):
        """Test outcome of valid and invalid *chain* arguments."""

        path = pathDatafile(self.multi['file'])
        self.assertRaises(TypeError, parseMMCIF, path, chain=['A'])
        self.assertRaises(ValueError, parseMMCIF, path, chain='')
        self.assertIsNone(parseMMCIF(path, chain='$'))
        self.assertEqual(parseMMCIF(path, chain='A').numAtoms(), 
                        self.multi['chainA_atoms'],
                        'parseMMCIF failed to parse correct number of atoms '
                        'when chain is specified')

    def testLongChainArgument(self):
        """Test outcome of valid and invalid *segment* arguments."""

        path = pathDatafile(self.no_pdb['file'])
        self.assertRaises(TypeError, parseMMCIF, path, segment=['SX0'])
        self.assertRaises(ValueError, parseMMCIF, path, segment='')
        self.assertIsNone(parseMMCIF(path, segment='$'))
        self.assertEqual(parseMMCIF(path, segment='SX0').numAtoms(), 
                        self.no_pdb['segment_SX0_atoms'],
                        'parseMMCIF failed to parse correct number of atoms '
                        'when segment SX0 is specified')

    def testSubsetArgument(self):
        """Test outcome of valid and invalid *subset* arguments."""

        path = pathDatafile(self.multi['file'])
        self.assertRaises(TypeError, parseMMCIF, path, subset=['A'])
        self.assertEqual(parseMMCIF(path, subset='ca').numAtoms(), 
                        self.multi['ca_atoms'],
                        'failed to parse correct number of "ca" atoms')
        self.assertEqual(parseMMCIF(path, subset='bb').numAtoms(),  
                        self.multi['bb_atoms'],
                        'failed to parse correct number of "bb" atoms')

    def testAgArgument(self):
        """Test outcome of valid and invalid *ag* arguments."""

        path = pathDatafile(self.multi['file'])
        self.assertRaises(TypeError, parseMMCIF, path, ag='AtomGroup')
        ag = prody.AtomGroup('One atom')
        ag.setCoords(np.array([[0., 0., 0.]]))
        self.assertRaises(ValueError, parseMMCIF, path, ag=ag)
        ag = prody.AtomGroup('Test')
        self.assertEqual(parseMMCIF(path, ag=ag).numAtoms(),
            self.multi['atoms'],
            'parseMMCIF failed to parse correct number of atoms')

    def testAgArgMultiModel(self):
        """Test number of coordinate sets when using *ag* arguments."""

        path = pathDatafile(self.multi['file'])
        ag = parseMMCIF(path)
        coords = ag.getCoordsets()
        ncsets = ag.numCoordsets()
        ag = parseMMCIF(path, ag=ag)
        self.assertEqual(ag.numCoordsets(), ncsets*2,
            'parseMMCIF failed to append coordinate sets to given ag')
        assert_equal(coords, ag.getCoordsets(np.arange(ncsets, ncsets*2)))


