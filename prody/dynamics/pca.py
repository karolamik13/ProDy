# -*- coding: utf-8 -*-
"""This module defines classes for principal component analysis (PCA) and
essential dynamics analysis (EDA) calculations."""

import time
import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from prody import LOGGER, PY2K
from prody.atomic import Atomic
from prody.ensemble import Ensemble, PDBEnsemble
from prody.trajectory import TrajBase
from prody.utilities import importLA, ZERO, solveEig

from .nma import NMA


if PY2K:
    range = xrange

__all__ = ['PCA', 'EDA']

class PCA(NMA):
    def __init__(self, name):
        super().__init__(name)
        self._cov = None
        self._eigvals = None
        self._array = None
        self._vars = None
        self._n_modes = None


    def setCovariance(self, covariance, is3d=True):
        """Set covariance matrix."""
        if not isinstance(covariance, np.ndarray):
            raise TypeError('covariance must be an ndarray')
        elif not (covariance.ndim == 2 and covariance.shape[0] == covariance.shape[1]):
            raise ValueError('covariance must be square matrix')
        elif covariance.dtype != float:
            try:
                covariance = covariance.astype(float)
            except:
                raise ValueError('covariance.dtype must be float')

        self._reset()

        self._is3d = is3d
        self._cov = covariance
        self._dof = covariance.shape[0]

        if is3d:
            self._n_atoms = self._dof // 3
        else:
            self._n_atoms = self._dof
        self._trace = self._cov.trace()

    def buildCovariance(self, coordsets, **kwargs):
        """Build a covariance matrix for *coordsets* using mean coordinates
        as the reference.  *coordsets* argument may be one of the following:

        * :class:`.Atomic`
        * :class:`.Ensemble`
        * :class:`.TrajBase`
        * :class:`numpy.ndarray` with shape ``(n_csets, n_atoms, 3)``

        For ensemble and trajectory objects, ``update_coords=True`` argument
        can be used to set the mean coordinates as the coordinates of the
        object.

        When *coordsets* is a trajectory object, such as :class:`.DCDFile`,
        covariance will be built by superposing frames onto the reference
        coordinate set (see :meth:`.Frame.superpose`).  If frames are already
        aligned, use ``aligned=True`` argument to skip this step.

        .. note::
           If *coordsets* is a :class:`.PDBEnsemble` instance, coordinates are
           treated specially.  Let's say **C**\_ij is the element of the
           covariance matrix that corresponds to atoms *i* and *j*.  This
           super element is divided by number of coordinate sets (PDB models or
           structures) in which both of these atoms are observed together."""

        if not isinstance(coordsets, (Ensemble, Atomic, TrajBase, np.ndarray)):
            raise TypeError('coordsets must be an Ensemble, Atomic, Numpy '
                            'array instance')
        LOGGER.timeit('_prody_pca')
        mean = None
        weights = None
        ensemble = None
        if isinstance(coordsets, np.ndarray):
            if (coordsets.ndim != 3 or coordsets.shape[2] != 3 or
                    coordsets.dtype not in (np.float32, float)):
                raise ValueError('coordsets is not a valid coordinate array')
        elif isinstance(coordsets, Atomic):
            coordsets = coordsets._getCoordsets()
        elif isinstance(coordsets, Ensemble):
            ensemble = coordsets
            if isinstance(coordsets, PDBEnsemble):
                weights = coordsets.getWeights() > 0
            coordsets = coordsets._getCoordsets()

        update_coords = bool(kwargs.get('update_coords', False))
        quiet = kwargs.pop('quiet', False)

        if isinstance(coordsets, TrajBase):
            nfi = coordsets.nextIndex()
            coordsets.reset()
            n_atoms = coordsets.numSelected()
            dof = n_atoms * 3
            cov = np.zeros((dof, dof))
            n_confs = 0
            n_frames = len(coordsets)
            if not quiet:
                LOGGER.info('Covariance will be calculated using {0} frames.'
                            .format(n_frames))
            coordsum = np.zeros(dof)
            if not quiet:
                LOGGER.progress('Building covariance', n_frames, '_prody_pca')
            align = not kwargs.get('aligned', False)
            for frame in coordsets:
                if align:
                    frame.superpose()
                coords = frame._getCoords().flatten()
                coordsum += coords
                cov += np.outer(coords, coords)
                n_confs += 1
                if not quiet:
                    LOGGER.update(n_confs, label='_prody_pca')
            if not quiet:
                LOGGER.finish()
            cov /= n_confs
            coordsum /= n_confs
            mean = coordsum
            cov -= np.outer(coordsum, coordsum)
            coordsets.goto(nfi)
            self._cov = cov
            if update_coords:
                coordsets.setCoords(mean.reshape((n_atoms, 3)))
        else:
            n_confs = coordsets.shape[0]
            if n_confs < 3:
                raise ValueError('coordsets must have more than 3 coordinate '
                                 'sets')
            n_atoms = coordsets.shape[1]
            if n_atoms < 3:
                raise ValueError('coordsets must have more than 3 atoms')
            dof = n_atoms * 3
            if not quiet:
                LOGGER.info('Covariance is calculated using {0} coordinate sets.'
                            .format(len(coordsets)))
            s = (n_confs, dof)
            if weights is None:
                if coordsets.dtype == float:
                    self._cov = np.cov(coordsets.reshape((n_confs, dof)).T,
                                       bias=1)
                else:
                    cov = np.zeros((dof, dof))
                    coordsets = coordsets.reshape((n_confs, dof))
                    mean = coordsets.mean(0)
                    if not quiet:
                        LOGGER.progress('Building covariance', n_confs,
                                        '_prody_pca')
                    for i, coords in enumerate(coordsets.reshape(s)):
                        deviations = coords - mean
                        cov += np.outer(deviations, deviations)
                        if not quiet:
                            LOGGER.update(n_confs, label='_prody_pca')
                    if not quiet:
                        LOGGER.finish()
                    cov /= n_confs
                    self._cov = cov
            else:
                # PDB ensemble case
                mean = np.zeros((n_atoms, 3))
                for i, coords in enumerate(coordsets):
                    mean += coords * weights[i]
                mean /= weights.sum(0)
                d_xyz = ((coordsets - mean) * weights).reshape(s)
                divide_by = weights.astype(float).repeat(3, axis=2).reshape(s)
                self._cov = np.dot(d_xyz.T, d_xyz) / np.dot(divide_by.T,
                                                            divide_by)
                if update_coords and ensemble is not None:
                    if mean is None:
                        mean = coordsets.mean(0)
                    ensemble.setCoords(mean)

        self._trace = self._cov.trace()
        self._dof = dof
        self._n_atoms = n_atoms
        if not quiet:
            LOGGER.report('Covariance matrix calculated in %2fs.', '_prody_pca')

    def performSVD(self, coordsets):
        """Calculate principal modes using singular value decomposition (SVD).
        *coordsets* argument may be a :class:`.Atomic`, :class:`.Ensemble`,
        or :class:`numpy.ndarray` instance.  If *coordsets* is a numpy array,
        its shape must be ``(n_csets, n_atoms, 3)``.  Note that coordinate
        sets must be aligned prior to SVD calculations.

        This is a considerably faster way of performing PCA calculations
        compared to eigenvalue decomposition of covariance matrix, but is
        an approximate method when heterogeneous datasets are analyzed.
        Covariance method should be preferred over this one for analysis of
        ensembles with missing atomic data.  See :ref:`pca-xray-calculations`
        example for comparison of results from SVD and covariance methods."""

        linalg = importLA()

        start = time.time()
        if not isinstance(coordsets, (Ensemble, Atomic, np.ndarray)):
            raise TypeError('coordsets must be an Ensemble, Atomic, Numpy '
                            'array instance')
        if isinstance(coordsets, np.ndarray):
            if (coordsets.ndim != 3 or coordsets.shape[2] != 3 or
                    coordsets.dtype not in (np.float32, float)):
                raise ValueError('coordsets is not a valid coordinate array')
            deviations = coordsets - coordsets.mean(0)
        else:
            if isinstance(coordsets, Ensemble):
                deviations = coordsets.getDeviations()
            elif isinstance(coordsets, Atomic):
                deviations = (coordsets._getCoordsets() -
                              coordsets._getCoords())

        n_confs = deviations.shape[0]
        if n_confs <= 3:
            raise ValueError('coordsets must have more than 3 coordinate sets')
        n_atoms = deviations.shape[1]
        if n_atoms <= 3:
            raise ValueError('coordsets must have more than 3 atoms')

        dof = n_atoms * 3
        deviations = deviations.reshape((n_confs, dof)).T

        vectors, values, self._temp = linalg.svd(deviations,
                                                 full_matrices=False)
        values = (values ** 2) / n_confs
        self._dof = dof
        self._n_atoms = n_atoms
        which = values > 1e-18
        self._eigvals = values[which]
        self._array = vectors[:, which]
        self._vars = self._eigvals
        self._trace = self._vars.sum()
        self._n_modes = len(self._eigvals)
        LOGGER.debug('{0} modes were calculated in {1:.2f}s.'
                     .format(self._n_modes, time.time()-start))

    def addEigenpair(self, eigenvector, eigenvalue=None):
        """Add eigen *vector* and eigen *value* pair(s) to the instance.
        If eigen *value* is omitted, it will be set to 1.  Eigenvalues
        are set as variances."""

        super().addEigenpair(self, eigenvector, eigenvalue)
        self._vars = self._eigvals

    def setEigens(self, vectors, values=None):
        """Set eigen *vectors* and eigen *values*.  If eigen *values* are
        omitted, they will be set to 1.  Eigenvalues are set as variances."""

        self._clear()
        super().setEigens(self, vectors, values)
        self._vars = self._eigvals
        

    def computeKernelMatrix(self, coordsets, kernel='rbf', degree=3, gamma=None, alpha=1.0, coef0=0.0):
        """Compute the kernel matrix for Kernel PCA."""
        
        LOGGER.info("Computing Kernel Matrix")
        if isinstance(coordsets, np.ndarray):
            coordsets = coordsets.reshape((coordsets.shape[0], -1))
        elif isinstance(coordsets, Ensemble):
            coordsets = coordsets.getCoordsets().reshape((coordsets.numConfs(), -1))
        else:
            raise TypeError('X must be a numpy array or Ensemble')
            
        if kernel == 'linear':
            return np.dot(coordsets, coordsets.T)
        elif kernel == 'poly':
            return (np.dot(coordsets, coordsets.T) + 1) ** degree
        elif kernel == 'rbf':
            if gamma is None:
                gamma = 1.0 / coordsets.shape[1]
            sq_dists = pdist(coordsets, 'sqeuclidean')
            mat_sq_dists = squareform(sq_dists)
            return np.exp(-gamma * mat_sq_dists)
        elif kernel == 'sigmoid':
            return np.tanh(alpha * np.dot(coordsets, coordsets.T) + coef0)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        LOGGER.info("Kernel Matrix computed")

    def calcModes(self, n_modes=20, turbo=True, kernel=None, degree=3, gamma=None, alpha=1.0, coef0=0.0, cv=5, max_iter=500, solver='lbfsg', **kwargs):
        """Calculate principal (or essential) modes using Kernel PCA if a kernel is specified."""
        
        if kernel:
            LOGGER.info("Performing Kernel PCA")
            if self._cov is None:
                raise ValueError('covariance matrix is not built or set')
            start = time.time()
            self._clear()
            if str(n_modes).lower() == 'all':
                n_modes = None

            #Create artifical labels using KMeans
            labels = self.createArtificialLabels(self._cov)
            
            # Optimize kernel PCA parameters
            if kernel in ['rbf', 'poly', 'sigmoid']:
                best_params = self.optimizeKernelPCAParams(kernel=kernel, n_modes=n_modes, cv=cv, labels=labels, max_iter=max_iter)
                degree = best_params.get('kpca__degree', degree)
                gamma = best_params.get('kpca__gamma', gamma)
                alpha = best_params.get('kpca__alpha', alpha)
                coef0 = best_params.get('kpca__coef0', coef0)
                
            # Compute the kernel matrix
            K = self.computeKernelMatrix(self._cov, kernel=kernel, degree=degree, gamma=gamma, alpha=alpha, coef0=coef0)

            # Center the kernel matrix
            N = K.shape[0]
            one_n = np.ones((N, N)) / N
            K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

            # Solve the eigenvalue problem for the centered kernel matrix
            values, vectors = eigh(K_centered)
            which = values > ZERO
            self._eigvals = values[which]
            self._array = vectors[:, which]
            self._vars = values[which]
            self._n_modes = len(self._eigvals)
            if self._n_modes > 1:
                LOGGER.debug('{0} modes were calculated in {1:.2f}s.'
                            .format(self._n_modes, time.time()-start))
            else:
                LOGGER.debug('{0} mode was calculated in {1:.2f}s.'
                            .format(self._n_modes, time.time()-start))
        else:
            # Fallback to original PCA calculation if no kernel is specified
            LOGGER.info("Standard PCA path")
            if self._cov is None:
                raise ValueError('covariance matrix is not built or set')
            start = time.time()
            self._clear()
            if str(n_modes).lower() == 'all':
                n_modes = None
            
            values, vectors, _ = solveEig(self._cov, n_modes=n_modes, zeros=True, 
                                          turbo=turbo, reverse=True, **kwargs)
            which = values > ZERO
            self._eigvals = values[which]
            self._array = vectors[:, which]
            self._vars = values[which]
            self._n_modes = len(self._eigvals)
            if self._n_modes > 1:
                LOGGER.debug('{0} modes were calculated in {1:.2f}s.'
                        .format(self._n_modes, time.time()-start))
            else:
                LOGGER.debug('{0} mode was calculated in {1:.2f}s.'
                        .format(self._n_modes, time.time()-start))

    
    def findBestSolver(self, max_iter=500, cv=5, labels=None):
    
        log_reg = LogisticRegression(max_iter=max_iter)
        
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', log_reg)
        ])
        
        param_grid = {
            'log_reg__solver': ['lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg']
        }
        
        LOGGER.info("Finding the best solver")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            grid_search = GridSearchCV(clf, param_grid, cv=cv)
            grid_search.fit(self._cov, labels)
        
        best_solver = grid_search.best_params_['log_reg__solver']
        LOGGER.info(f"Best solver found: {best_solver}")
        
        return best_solver 
    
    def optimizeKernelPCAParams(self, kernel='rbf', n_modes=10, cv=5, labels=None, max_iter=500, **kwargs):
        
        best_solver = self.findBestSolver(max_iter=max_iter, cv=cv, labels=labels)
        
        if kernel == 'poly':
            param_grid = {
                'kpca__degree': np.linspace(1, 6, 6)
            }
        elif kernel == 'rbf':
            param_grid = {
                'kpca__gamma': np.linspace(0.03, 0.06, 20)
            }
        elif kernel == 'sigmoid':
            param_grid = {
                'kpca__alpha': np.linspace(0.02, 1.00, 50) ,
                'kpca__coef0': np.linspace(0.00, 10, 50)
            }
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")


        log_reg = LogisticRegression(max_iter=max_iter, solver=best_solver)
        
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("kpca", KernelPCA(n_components=n_modes, kernel=kernel)),
            ("log_reg", log_reg)])

        LOGGER.info("Pipeline clf created")

        grid_search = GridSearchCV(clf, param_grid, cv=cv)
        grid_search.fit(self._cov, labels)
        
        # Log mean test score for each solver
        cv_results = grid_search.cv_results_
        LOGGER.info("Mean test scores:")
        for params, mean_score in zip(cv_results['params'], cv_results['mean_test_score']):
            LOGGER.info(f"Solver: {params['log_reg__solver']}, Mean score: {mean_score:.4f}")

        best_params = grid_search.best_params_
        LOGGER.info(f"Best parameters found for kernel {kernel}: {best_params}")

        return best_params
                
                
    def createArtificialLabels(self, coordsets, n_clusters=2):
        """Create artificial labels using KMeans clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coordsets)
        return kmeans.labels_
        
                
    def _clear(self):
        """Clear attributes."""
        self._eigvals = None
        self._array = None
        self._vars = None
        self._n_modes = None

    def _report(self):
        """Report calculation results."""
        pass  # Replace with actual reporting functionality


class EDA(PCA):

    """A class for Essential Dynamics Analysis (EDA) [AA93]_.
    See examples in :ref:`eda`.

    .. [AA93] Amadei A, Linssen AB, Berendsen HJ. Essential dynamics of
       proteins. *Proteins* **1993** 17(4):412-25."""

    pass
