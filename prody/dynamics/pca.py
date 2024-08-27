# -*- coding: utf-8 -*-
"""This module defines classes for principal component analysis (PCA) and
essential dynamics analysis (EDA) calculations."""

import time
import numpy as np
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from prody import LOGGER
from prody.atomic import Atomic
from prody.ensemble import Ensemble, PDBEnsemble
from prody.trajectory import TrajBase
from prody.utilities import importLA, ZERO, solveEig

from .nma import NMA


__all__ = ['PCA', 'EDA']

class PCA(NMA):
    def __init__(self, name='Unknown'):
        super().__init__(name)
        self._eigvals = None
        self._array = None
        self._vars = None
        self._n_modes = None
        
        NMA.__init__(self, name)

    
    def computeKernelMatrix(self, coordsets, kernel='rbf', degree=3, gamma=None, alpha=1.0, coef0=0.0):
        """By specifying any Kernel when calling the calcModes function
        additionally an adequate matrix has to be calculated to perform
        Kernel PCA. This function computes the kernel matrix for Kernel PCA.
        For each kernel the appropriate equation with required parameters
        is called.
        
        linear: no additional parameter required
        poly: degree (default = 3), recommended 2-6
        rbf: gamma (default = None, integreted calculation using coordsets.shape), recommended 0.3-0.8
        sigmoid: alpha, coef0 (default = 1.0 and 0.0), recommended 0.2-1.0 and 0-10"""
        
        LOGGER.info("Computing Kernel Matrix")  
        
        #LOGGER.info(f"{coordsets.ndim}") 
        
        if coordsets.ndim == 1:
            coordsets = coordsets.reshape(-1, 1)  
            
        #LOGGER.info(f"{coordsets}")     
            
        if kernel == 'linear':
            K = np.dot(coordsets, coordsets.T)
            #LOGGER.info(f"{K}") 
        elif kernel == 'poly':
            K = (np.dot(coordsets, coordsets.T) + 1) ** degree
            #LOGGER.info(f"{K}")
        elif kernel == 'rbf':
            if gamma is None:
                gamma = 1.0 / coordsets.shape[1]
            sq_dists = pdist(coordsets, 'sqeuclidean')
            mat_sq_dists = squareform(sq_dists)
            K = np.exp(-gamma * mat_sq_dists)
            #LOGGER.info(f"{K}")
        elif kernel == 'sigmoid':
            K = np.tanh(alpha * np.dot(coordsets, coordsets.T) + coef0)
            #LOGGER.info(f"{K}")
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        LOGGER.info("Kernel Matrix computed")
        
        return K


    def createArtificialLabels(self, coordsets, kernel='rbf', degree=3, gamma=None, alpha=1.0, coef0=0.0, n_clusters=2):
        """Create artificial labels using KMeans clustering.
        necessary for Cross Validation process;
        else it returns an error"""
        
        K = self.computeKernelMatrix(coordsets, kernel=kernel, degree=degree, gamma=gamma, alpha=alpha, coef0=coef0)
        LOGGER.info(f"{K}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(K)
        LOGGER.info(f"{kmeans}")
        return kmeans.labels_
    
    def calcModes(self, coordsets, n_modes=20, turbo=True, kernel='rbf', degree=3, gamma=None, alpha=1.0, coef0=0.0, optimize_params=False, best_kernel=False, max_iter=500, labels=None, **kwargs):
        """Calculate principal (or essential) modes using Kernel PCA with a kernel specified.
        
        linear: no additional parameter required
        poly: degree (default = 3), recommended 2-6
        rbf: gamma (default = None, integreted calculation using coordsets.shape), recommended 0.3-0.8
        sigmoid: alpha, coef0 (default = 1.0 and 0.0), recommended 0.2-1.0 and 0-10"""
        
        LOGGER.timeit('_prody_pca')
        start = time.time()
        
        # check if coordsets are either Ensemble, Atomic, TrajBase or np.ndarray 
        if not isinstance(coordsets, (Ensemble, Atomic, TrajBase, np.ndarray)):
            raise TypeError('coordsets must be an Ensemble, Atomic, Numpy array instance')
        
        # Check if coordsets has the correct shape and extract coordinates according to its features
        if isinstance(coordsets, np.ndarray):
            if (coordsets.ndim != 3 or coordsets.shape[2] != 3 or coordsets.dtype not in (np.float32, float)):
                raise ValueError('coordsets is not a valid coordinate array')          
        elif isinstance(coordsets, Atomic):
            coordsets = coordsets._getCoordsets()
        elif isinstance(coordsets, Ensemble):
            self._atoms = coordsets.getAtoms()
            LOGGER.debug(f"Atom set: {self._atoms}")
            if isinstance(coordsets, PDBEnsemble):
                weights = coordsets.getWeights() > 0
            coordsets = coordsets._getCoordsets()
            if coordsets.ndim != 3 or coordsets.shape[2] != 3:
                raise ValueError('coordsets should have shape (n_frames, n_atoms, 3)')

        # reshape coordsets to fit the required shape if necessary
        coordsets = coordsets.reshape((coordsets.shape[0], -1))
        n_atoms_from_coordsets = coordsets.shape[1] // 3
        
        # debug message to check if the coordinates are parsed correctly
        if self._atoms is not None:
            num_atoms_from_ensemble = len(self._atoms)
            LOGGER.debug(f"Number of atoms from ensemble: {num_atoms_from_ensemble}")
            if n_atoms_from_coordsets != num_atoms_from_ensemble:
                raise ValueError(f'Number of atoms in ensemble ({num_atoms_from_ensemble}) does not match coordsets ({n_atoms_from_coordsets})')
        
        # Create artifical labels using KMeans for Logistic Regression
        if labels is None:
             labels = self.createArtificialLabels(coordsets, kernel=kernel, degree=degree, gamma=gamma, alpha=alpha, coef0=coef0)       
        
        # optimize parameters if optimize_params=True   
        if optimize_params==True:
            # Optimize kernel PCA parameters when chosing a kernel manually
            if kernel in ['rbf', 'poly', 'sigmoid'] and any(param is None for param in [degree, gamma, alpha, coef0]):
                best_params = self.optimizeKernelPCAParams(coordsets, kernel=kernel, n_modes=n_modes, max_iter=max_iter, labels=labels)
                degree = best_params.get('kpca__degree', degree)
                gamma = best_params.get('kpca__gamma', gamma)
                alpha = best_params.get('kpca__alpha', alpha)
                coef0 = best_params.get('kpca__coef0', coef0) 
        
        # Compute the kernel matrix
        K = self.computeKernelMatrix(coordsets, kernel=kernel, degree=degree, gamma=gamma, alpha=alpha, coef0=coef0)
        #LOGGER.info(f"{np.array(K)}")

        # Center the kernel matrix
        N = K.shape[0]
        #LOGGER.info(f"{N}")
        one_n = np.ones((N, N)) / N
        #LOGGER.info(f"{one_n}")
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        #LOGGER.info(f"{K_centered}")

        # Solve the eigenvalue problem for the centered kernel matrix
        #values, vectors = eigh(K_centered)
        values, vectors, _ = solveEig(K_centered, n_modes=n_modes, zeros=True, turbo=turbo, reverse=True, **kwargs) 
        #LOGGER.info(f"{values}")
        #LOGGER.info(f"{vectors}")
            
        # Filter very small and negative eigenvalues
        # ZERO = 1e-10	# treshold can be hardcoded here
        which = values > ZERO
        #LOGGER.info(f"{which}")
        values = values[which]
        #LOGGER.info(f"{values}")
        vectors = vectors[:, which]
        #LOGGER.info(f"{vectors}")
            
        # Select the top n_modes eigenvalues and eigenvectors
        # not sure if this makes any difference for other systems, it did not for the ones I tested
        indices = np.argsort(values)[::-1][:n_modes]
        #LOGGER.info(f"{indices}")
        values = values[indices]
        #LOGGER.info(f"{values}")
        vectors = vectors[:, indices]
        #LOGGER.info(f"{vectors}")

        # assign necessary data
        self._eigvals = values
        #LOGGER.info(f"{self._eigvals}")
        self._array = vectors
        #LOGGER.info(f"{self._array}")
        self._vars = values
        #LOGGER.info(f"{self._eigvals}")
        self._n_modes = len(values)
        #LOGGER.info(f"{self._n_modes}")

        if self._n_modes > 1:
            LOGGER.debug('{0} modes were calculated in {1:.2f}s.'.format(self._n_modes, time.time()-start))
        else:
            LOGGER.debug('{0} mode was calculated in {1:.2f}s.'.format(self._n_modes, time.time()-start))

        #LOGGER.debug(f'PCA eigenvalues shape: {self._eigvals.shape}')
        #LOGGER.debug(f'PCA array shape: {self._array.shape}')
        
        # Ensure the results are in the expected format
        if self._array.shape[1] != n_modes:
            raise ValueError(f'Number of modes in PCA array ({self._array.shape[1]}) does not match expected ({n_modes})')

  
    def optimizeKernelPCAParams(self, coordsets, kernel='rbf', n_modes=10, cv=5, max_iter=500, solver='sag', labels=None, **kwargs):
        """This function is used to determine the best parameters
        for each Kernel of the Kernel PCA approach using a Pipeline
        consisting of a Standard Scaler to scale the data accordingly,
        an implemented Kernel PCA from the sklearn package and
        a linear Regression model with the predefined 'newton-cg' solver.
        The Cross Validation to determine the best parameters is based
        on a grid-search algorithm which relies on a K-Mean validation process.
        
        Output: best parameters for each kernel (default: rbf)
        """
        
        if kernel == 'poly':
            param_grid = {
                'kpca__degree': np.arange(1, 7)
            }
            LOGGER.info(f"{param_grid}")
        elif kernel == 'linear':
            param_grid = {}
            LOGGER.info(f"{param_grid}")
        elif kernel == 'rbf':
            param_grid = {
                'kpca__gamma': np.logspace(-3, 3, 7)
            }
            LOGGER.info(f"{param_grid}")
        elif kernel == 'sigmoid':
            param_grid = {
                'kpca__alpha': np.linspace(0.02, 1.00, 10) ,
                'kpca__coef0': np.linspace(0.00, 10, 10)
            }
            LOGGER.info(f"{param_grid}")
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        # create the logistic regression calculation
        log_reg = LogisticRegression(max_iter=max_iter, solver=solver)
        LOGGER.info(f"{log_reg}")
        
        # create the cross validation pipeline
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("kpca", KernelPCA(n_components=n_modes, kernel=kernel)),
            ("log_reg", log_reg)])
        LOGGER.info(f"{clf}")

        #LOGGER.info("Pipeline clf created")
        
        # run cross validation of the parameters
        if param_grid:
            grid_search = GridSearchCV(clf, param_grid, cv=cv)
            grid_search.fit(coordsets, labels)
            best_params = grid_search.best_params_
            LOGGER.info(f"Best parameters found for kernel {kernel}: {best_params}")
        else:
            best_params = {}
            LOGGER.info(f"No parameter tuning required for kernel {kernel}")
            
        return best_params
                
        
    ## not implemented in the calculation yet   
    def BestKernel(self, n_modes=10, cv=5, max_iter=500, solver='newton-cg', **kwargs):
        """Use this function to determine the best Kernel for your dataset
        solver set to 'newton-cg' on default again
        sigmoid runs a very long time; consider not using this one
        """
        
        LOGGER.info("Testing the best Kernel for Kernel PCA")
        
        # Define Kernel types
        kernels = ['linear', 'poly', 'rbf', 'sigmoid'] # hardcoded here, more could be added, or deleted but then need to adapt also all other functions
        best_kernel = None
        best_params = None
        best_score = -np.inf

        
        # Test Kernels
        for kernel in kernels:
            LOGGER.info(f"Testing kernel: {kernel}")
            
            # Optimize kernel PCA parameters for current kernel and create pipeline with Kernel PCA and Logistic Regression
            params = self.optimizeKernelPCAParams(coordsets, kernel=kernel, n_modes=n_modes, cv=cv, max_iter=max_iter)
            
            if kernel == 'linear':
                kpca_params = {}
            else:
                kpca_params = {key[len('kpca__'):]: value for key, value in params.items() if key.startswith('kpca__')}
            
            kpca = KernelPCA(n_components=n_modes, kernel=kernel, **kpca_params)
            log_reg = LogisticRegression(max_iter=max_iter, solver=solver)
            
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("kpca", kpca),
                ("log_reg", log_reg)
            ])
            
            scores = cross_val_score(clf, coordsets, cv=cv, scoring='accuracy', n_jobs=-1) 
            score = scores.mean()

            LOGGER.info(f"Kernel: {kernel}, Mean CV Score: {mean_score:.4f}")
            
            # Store best kernel and keep it updated
            if score > best_score:
                best_score = score
                best_kernel = kernel
                best_params = params
        
        LOGGER.info(f"Best kernel: {best_kernel} with a score of: {best_score:.4f}")
        LOGGER.info(f"Best parameters: {best_params}")
        
        return best_kernel, best_params 
        
        # Standard PCA script starts here
        
    def setCovariance(self, covariance, is3d=True):
        """Set covariance matrix."""

        if not isinstance(covariance, np.ndarray):
            raise TypeError('covariance must be an ndarray')
        elif not (covariance.ndim == 2 and
                  covariance.shape[0] == covariance.shape[1]):
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
            #mean = coordsets._getCoords().flatten()
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
            
            
    def calcModesPCA(self, n_modes=20, turbo=True, **kwargs):
        """Calculate principal (or essential) modes.  This method uses
        :func:`scipy.linalg.eigh`, or :func:`numpy.linalg.eigh`, function
        to diagonalize the covariance matrix.

        :arg n_modes: number of non-zero eigenvalues/vectors to calculate,
            default is 20,
            if **None** or ``'all'`` is given, all modes will be calculated
        :type n_modes: int

        :arg turbo: when available, use a memory intensive but faster way to
            calculate modes, default is **True**
        :type turbo: bool"""
        
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

        NMA.addEigenpair(self, eigenvector, eigenvalue)
        self._vars = self._eigvals

    
    def setEigens(self, vectors, values=None):
        """Set eigen *vectors* and eigen *values*.  If eigen *values* are
        omitted, they will be set to 1.  Eigenvalues are set as variances."""

        self._clear()
        NMA.setEigens(self, vectors, values)
        self._vars = self._eigvals
    
    
    def _clear(self):
        """Clear attributes."""
        self._eigvals = None
        self._array = None
        self._vars = None
        self._n_modes = None
        
    



class EDA(PCA):

    """A class for Essential Dynamics Analysis (EDA) [AA93]_.
    See examples in :ref:`eda`.

    .. [AA93] Amadei A, Linssen AB, Berendsen HJ. Essential dynamics of
       proteins. *Proteins* **1993** 17(4):412-25."""

    pass
