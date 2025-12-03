import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.special import hermite
from math import factorial
from debugger import message_checkpoints, matrix_checkpoints, vector_checkpoints

class HarmonicOscillator:
    """
    Harmonic oscillator class handling operators, eigenvalue problems, and error metrics.

    Args:
        a (float): Left boundary of spatial domain.
        b (float): Right boundary of spatial domain.
        npts (int): Number of discretization points.
        m (float): Mass of the particle.
        omega (float): Oscillator frequency.
        hbar (float): Reduced Planck constant.
        derivative_order (int): Finite difference order (2 or 4).
        sparse (bool): Whether to use sparse matrices.
        V_func (callable, optional): Custom potential function.

    Raises:
        AssertionError: If derivative_order is neither 2 nor 4.
    """

    def __init__(self, a=-10., b=10., npts=1001, m=1.0, omega=1.0, hbar=1.0, derivative_order=2, sparse=True, V_func=None):
        assert derivative_order in (2, 4), "derivative_order must be 2 or 4!"
        self.a = float(a)
        self.b = float(b)
        self.npts = int(npts)
        self.m = float(m)
        self.omega = float(omega)
        self.hbar = float(hbar)
        self.derivative_order = int(derivative_order)
        self.dx = np.abs(self.b - self.a) / (self.npts - 1)
        self.x = np.linspace(self.a, self.b, self.npts)
        self.sparse = bool(sparse)

        if V_func is None:
            self.V_func = lambda x: 0.5 * self.m * self.omega**2 * x**2
        else:
            self.V_func = V_func

        self._H = None
        self._cached_k = 0
        self._cached_evals = None
        self._cached_evecs = None

        self._build_hamiltonian()

    def potential_operator(self):
        """
        Compute the potential energy operator.

        Returns:
            csr_matrix: Diagonal sparse matrix representing the potential.
        """
        Vvals = self.V_func(self.x)
        return diags(Vvals, 0, format="csr")

    def kinetic_operator(self):
        """
        Compute the kinetic operator using finite differences.

        Raises:
            TypeError: If derivative_order is not 2 or 4.

        Returns:
            csr_matrix: Sparse matrix representation of the kinetic operator.
        """
        if self.sparse:
            if self.derivative_order == 2:
                diag_el = 2 * np.ones(self.npts)
                offdiag_el = -1 * np.ones(self.npts - 1)
                K = diags([offdiag_el, diag_el, offdiag_el], [-1, 0, 1], format="csr")
            elif self.derivative_order == 4:
                diag_el = (5/2) * np.ones(self.npts)
                offdiag_el1 = -(4/3) * np.ones(self.npts - 1)
                offdiag_el2 = (1/12) * np.ones(self.npts - 2)
                K = diags([offdiag_el2, offdiag_el1, diag_el, offdiag_el1, offdiag_el2], offsets=[-2, -1, 0, 1, 2], format="csr")
            else:
                raise TypeError("Incorrect derivative order chosen. Choose 2 or 4.")

        K = 0.5 * self.hbar**2 * K / (self.m * self.dx**2)
        return K

    def _build_hamiltonian(self):
        """
        Construct Hamiltonian H = T + V and reset cached eigenpairs.

        Returns:
            csr_matrix: Hamiltonian matrix.
        """
        T = self.kinetic_operator()
        V = self.potential_operator()
        self._H = T + V
        self._cached_k = 0
        self._cached_evals = None
        self._cached_evecs = None
        return self._H

    @property
    def H(self):
        """
        Get Hamiltonian matrix, rebuilding if necessary.

        Returns:
            csr_matrix: Hamiltonian matrix.
        """
        if self._H is None:
            self._build_hamiltonian()
        return self._H

    def eigenproblem(self, k=10):
        """
        Solve for the k lowest eigenvalues and eigenvectors.

        Args:
            k (int): Number of eigenpairs to compute.

        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        if k <= self._cached_k and self._cached_evals is not None:
            return self._cached_evals[:k], self._cached_evecs[:, :k]

        evals, evecs = eigsh(self.H, k=k, which="SM")

        for i in range(k):
            norm = np.sqrt(np.sum(np.abs(evecs[:, i])**2) * self.dx)
            evecs[:, i] /= norm

        self._cached_k = k
        self._cached_evals = evals
        self._cached_evecs = evecs

        return evals, evecs

    def numerical_eigenvalues(self, k=10):
        """
        Compute numerical eigenvalues.

        Args:
            k (int): Number of eigenvalues.

        Returns:
            np.ndarray: Numerical eigenvalues.
        """
        evals, _ = self.eigenproblem(k)
        return evals

    def numerical_eigenvectors(self, k=10):
        """
        Compute normalized numerical eigenvectors.

        Args:
            k (int): Number of eigenvectors.

        Returns:
            np.ndarray: Array of eigenvectors.
        """
        _, evecs = self.eigenproblem(k)
        for i in range(k):
            vector_checkpoints(v=evecs[:, i], dx=self.dx, verbosity=0)
        return evecs

    def analytical_eigenvalues(self, k=10):
        """
        Analytical harmonic oscillator eigenvalues.

        Args:
            k (int): Number of eigenvalues.

        Returns:
            np.ndarray: Analytical eigenvalues.
        """
        k_values = np.arange(k)
        return self.hbar * self.omega * (k_values + 0.5)

    def analytical_eigenvectors(self, k=10, x=None):
        """
        Analytical eigenfunctions evaluated on grid.

        Args:
            k (int): Number of eigenfunctions.
            x (np.ndarray, optional): Grid points.

        Returns:
            np.ndarray: Matrix of eigenfunctions.
        """
        if x is None:
            x = self.x

        phi = []
        for kk in range(k):
            Hk = hermite(kk)
            Ck = np.sqrt(1.0 / (2**kk * factorial(kk))) * (self.m * self.omega / (np.pi * self.hbar))**0.25
            xi = np.sqrt((self.m * self.omega) / self.hbar) * x
            phi_k = Ck * Hk(xi) * np.exp(-0.5 * xi**2)

            if np.any(np.isnan(phi_k)):
                phi.append(np.zeros_like(self.x))
                continue

            norm = np.sqrt(np.sum(np.abs(phi_k)**2) * self.dx)
            if norm != 0:
                phi_k /= norm

            vector_checkpoints(0, phi_k, self.dx)
            phi.append(phi_k)

        return np.array(phi).T

    def eigenvalue_relative_errors(self, k=10):
        """
        Compute relative error between numerical and analytical eigenvalues.

        Args:
            k (int): Number of eigenvalues.

        Returns:
            np.ndarray: Relative errors.
        """
        numerical = self.numerical_eigenvalues(k)
        analytical = self.analytical_eigenvalues(k)
        return np.abs(numerical - analytical) / analytical

    def eigenvector_cosine_similarity(self, k=10):
        """
        Compute cosine similarity between numerical and analytical eigenvectors.

        Args:
            k (int): Number of eigenvectors.

        Returns:
            np.ndarray: Cosine similarities.
        """
        numerical = self.numerical_eigenvectors(k)
        analytical = self.analytical_eigenvectors(k)
        return np.abs(np.sum(numerical * analytical, axis=0) * self.dx)

    def eigenvector_cosine_errors(self, k=10):
        """
        Compute angular cosine error sin(theta).

        Args:
            k (int): Number of eigenvectors.

        Returns:
            np.ndarray: Cosine errors.
        """
        cos_sim = self.eigenvector_cosine_similarity(k)
        return np.sqrt(1.0 - cos_sim**2)