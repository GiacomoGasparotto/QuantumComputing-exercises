import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from math import factorial
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.special import hermite
from debugger import message_checkpoints, vector_checkpoints

class HarmonicOscillator:
    """
    Harmonic oscillator class handling operators, eigenvalue problems, and time evolution.

    This class provides methods to solve the time-independent and time-dependent 
    Schrodinger equation for a 1D harmonic oscillator using finite difference 
    and split-operator methods.
    """

    def __init__(self, 
                 a=-10., b=10., npts=1000, 
                 ti=0., tf=10., npts_t=1000,
                 simulation_time=10.,
                 m=1.0, omega=1.0, hbar=1.0, 
                 derivative_order=4, sparse=True,
                 time_dependent=False):
        """
        Initializes the HarmonicOscillator instance with simulation parameters.

        Args:
            a (float): Left boundary of spatial domain.
            b (float): Right boundary of spatial domain.
            npts (int): Number of spatial discretization points.
            ti (float): Initial time.
            tf (float): Final time parameter (used for potential scaling).
            npts_t (int): Number of time steps for the discretization.
            simulation_time (float): Total duration of the simulation.
            m (float): Mass of the particle.
            omega (float): Angular frequency of the oscillator.
            hbar (float): Reduced Planck constant.
            derivative_order (int): Order of the finite difference approximation (2 or 4).
            sparse (bool): If True, uses sparse matrix representations.
            time_dependent (bool): If True, enables time-dependent potential V(x,t).
        """
        
        assert derivative_order in (2, 4), "derivative_order must be 2 or 4!"

        # Space discretization parameters
        self.a = float(a)
        self.b = float(b)
        self.npts = int(npts)
        self.dx = np.abs(self.b - self.a) / (self.npts)
        self.x = np.linspace(self.a, self.b, self.npts)

        # Time discretization parameters
        self.ti = float(ti)
        self.tf = float(tf)
        self.npts_t = int(npts_t)
        self.dt = np.abs(self.tf - self.ti) / (self.npts_t)
        self.T = np.linspace(self.ti, self.tf, self.npts_t)

        # Simulation time
        self.simulation_time = simulation_time

        # Harmonic oscillator parameters
        self.m = float(m)
        self.omega = float(omega)
        self.hbar = float(hbar)
        self.derivative_order = int(derivative_order)

        # Momentum grid for FFT
        self.p_grid = 2 * np.pi * np.fft.fftfreq(self.npts, d=self.dx)
        self.P = self.hbar * diags(self.p_grid, 0, format="csr")
        
        self.sparse = bool(sparse)
        self.time_dependent = bool(time_dependent)

        self._H = None

    def potential_operator(self, t=None):
        """
        Computes the potential energy operator V(x) or V(x,t).

        Args:
            t (float, optional): Time variable. Required if time_dependent is True.

        Returns:
            scipy.sparse.csr_matrix or numpy.ndarray: Diagonal matrix (or array) representing the potential.
        
        Raises:
            TypeError: If time_dependent is True but t is not provided.
        """
        if not self.time_dependent:
            # Time-independent (static) potential
            Vvals = 0.5 * self.m * self.omega**2 * self.x**2
        else:
            # Time-dependent potential
            if t is None:
                raise TypeError("ERROR. Insert a time to use time dependent potential!")
            
            q0 = t/self.tf # potential velocity
            Vvals = 0.5 * self.m * self.omega**2 * (self.x - q0)**2
        
        if self.sparse:
            return diags(Vvals, 0, format="csr")
        else:
            return Vvals

    def kinetic_operator(self):
        """
        Computes the kinetic energy operator T using finite differences.

        Uses either a 2nd order or 4th order central difference scheme based on 
        self.derivative_order.

        Returns:
            scipy.sparse.csr_matrix: Sparse matrix representation of the kinetic operator.
        
        Raises:
            TypeError: If derivative_order is not 2 or 4.
        """
        if self.derivative_order == 2: # second order matrix
            diag_el = 2 * np.ones(self.npts)
            offdiag_el = -1 * np.ones(self.npts - 1)
            K = diags([offdiag_el, diag_el, offdiag_el], [-1, 0, 1], format="csr")

        elif self.derivative_order == 4: # fourth order matrix
            diag_el = (5/2) * np.ones(self.npts)
            offdiag_el1 = -(4/3) * np.ones(self.npts - 1)
            offdiag_el2 = (1/12) * np.ones(self.npts - 2)
            K = diags([offdiag_el2, offdiag_el1, diag_el, offdiag_el1, offdiag_el2], offsets=[-2, -1, 0, 1, 2], format="csr")

        else:
            raise TypeError("Incorrect derivative order chosen. Choose 2 or 4.")

        K = 0.5 * self.hbar**2 * K / (self.m * self.dx**2)
        return K

    def hamiltonian(self, t=None):
        """
        Constructs the full Hamiltonian matrix H = T + V.

        Args:
            t (float, optional): Time variable for time-dependent potentials.

        Returns:
            scipy.sparse.csr_matrix: Hamiltonian matrix.
        """
        K = self.kinetic_operator()
        V = self.potential_operator(t)
        return K + V

    @property
    def H(self):
        """
        Property to access the static Hamiltonian.

        Returns:
            scipy.sparse.csr_matrix: The static Hamiltonian matrix at t=0.
        
        Raises:
            RuntimeError: If the system is time-dependent (must use hamiltonian(t) instead).
        """
        if self.time_dependent:
            raise RuntimeError("Hamiltonian is time-dependent. Use hamiltonian(t).")
        
        if self._H is None:
            self._H = self.hamiltonian(t=0)
        
        return self._H

    def eigenproblem(self, k=10, t=0.):
        """
        Solves the eigenvalue problem H*psi = E*psi for the k lowest states.

        Args:
            k (int): Number of eigenvalues/eigenvectors to compute.
            t (float): Time at which to evaluate the Hamiltonian.

        Returns:
            tuple: A tuple (evals, evecs) where:
                - evals (numpy.ndarray): Array of k lowest eigenvalues.
                - evecs (numpy.ndarray): Matrix of shape (npts, k) containing eigenvectors.
        """
        H_t = self.hamiltonian(t)
        evals, evecs = eigsh(H_t, k=k, which="SM")

        for i in range(k):
            norm = np.sqrt(np.sum(np.abs(evecs[:, i])**2) * self.dx)
            evecs[:, i] /= norm

        return evals, evecs

    def numerical_eigenvalues(self, k=10, t=0.):
        """
        Computes numerical eigenvalues.

        Args:
            k (int): Number of eigenvalues to compute.
            t (float): Time parameter.

        Returns:
            numpy.ndarray: Array of eigenvalues.
        """
        evals, _ = self.eigenproblem(k, t)
        return evals

    def numerical_eigenvectors(self, k=10, t=0.):
        """
        Computes normalized numerical eigenvectors.

        Args:
            k (int): Number of eigenvectors to compute.
            t (float): Time parameter.

        Returns:
            numpy.ndarray: Matrix of eigenvectors.
        """
        _, evecs = self.eigenproblem(k, t)
        return evecs

    def analytical_eigenvalues(self, k=10):
        """
        Computes analytical harmonic oscillator eigenvalues.

        Args:
            k (int): Number of eigenvalues.

        Returns:
            numpy.ndarray: Array of analytical eigenvalues E_n = hbar*omega*(n + 0.5).
        """
        k_values = np.arange(k)
        return self.hbar * self.omega * (k_values + 0.5)

    def analytical_eigenvectors(self, k=10, x=None):
        """
        Computes analytical eigenfunctions (Hermite-Gaussian) evaluated on the grid.

        Args:
            k (int): Number of eigenfunctions.
            x (numpy.ndarray, optional): Spatial grid points. Defaults to self.x.

        Returns:
            numpy.ndarray: Matrix of analytical eigenfunctions.
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
                phi.append(np.zeros_like(x))
                continue

            norm = np.sqrt(np.sum(np.abs(phi_k)**2) * self.dx)
            if norm != 0:
                phi_k /= norm

            phi.append(phi_k)

        return np.array(phi).T

    # ========== SPLIT-OPERATOR METHOD ==========

    def time_evo(self, psi0, simulation_time=None):
        """
        Propagates an initial wavefunction in time using the Split-Step Fourier method.

        Args:
            psi0 (numpy.ndarray): Initial wavefunction array.
            simulation_time (float, optional): Duration of the simulation. Defaults to self.tf.

        Returns:
            tuple: A tuple (psi_time, T_grid) where:
                - psi_time (list): List of wavefunctions at each time step.
                - T_grid (numpy.ndarray): Array of time points corresponding to the steps.
        """
        if simulation_time is None:
            simulation_time = self.tf

        # initial value
        psi = psi0.copy()

        # Define grids
        T_grid = np.arange(self.ti, simulation_time, self.dt)
        p = self.p_grid * self.hbar

        # Save the entire time evolution
        psi_time = []  

        for t in T_grid:
            # Potential half step
            V = self.potential_operator(t).diagonal()
            U = np.exp(-1j * V * self.dt / (2*self.hbar))

            # Apply potential
            psi = U*psi
            vector_checkpoints(0, psi, self.dx) # normalization check

            # Fourier transform
            psi_p = np.fft.fft(psi)

            # Kinetic operator
            psi_p = np.exp(-1j * (p**2)/(2*self.m) * self.dt / self.hbar)*psi_p

            # Fourier antitrasform
            psi = np.fft.ifft(psi_p)

            # Apply potential
            psi = U*psi  
            vector_checkpoints(0, psi, self.dx) # normalization check     

            # Store time evolution
            psi_time.append(psi)

        return psi_time, T_grid

    def position_expectation_value(self, psi):
        """
        Calculates the expectation value of position <x>.

        Args:
            psi (numpy.ndarray): Wavefunction array (can be 1D or 2D array of time evolution).

        Returns:
            float or numpy.ndarray: The expectation value(s) of position.
        """
        return np.sum(np.abs(psi)**2 * self.x[None,:], axis=1) * self.dx
   
    def position_squared_expectation_value(self, psi):
        """
        Calculates the expectation value of position squared <x^2>.

        Args:
            psi (numpy.ndarray): Wavefunction array.

        Returns:
            float or numpy.ndarray: The expectation value(s) of x^2.
        """
        return np.sum(np.abs(psi)**2 * self.x[None,:]**2, axis=1) * self.dx

    def position_uncertainty(self, psi_t, exp_x):
        """
        Computes the position uncertainty (standard deviation) Delta x.

        Args:
            psi_t (numpy.ndarray): Wavefunction time evolution array.
            exp_x (numpy.ndarray): Array of position expectation values <x>.

        Returns:
            numpy.ndarray: Array of position uncertainties over time.
        """
        # <x^2>(t)
        x_sq = self.position_squared_expectation_value(psi_t)
        # (<x>(t))^2
        x_term_sq = exp_x**2
        
        # Uncertainty
        return np.sqrt(np.real(x_sq - x_term_sq))
    
    # ===============
    #      PLOTS
    # ===============

    def plot_time_evo(self, ax, psi_t, times_sec, transform_func, y_label, plot_legend=False):
        """
        Plots the time evolution of a transformed wavefunction on a specific Matplotlib axis.
        
        Args:
            ax (matplotlib.axes.Axes): The Matplotlib axis to draw on.
            psi_t (numpy.ndarray): Time evolution array (N_t, N_x).
            times_sec (list): List of desired times to plot.
            transform_func (callable): Function that transforms psi (e.g. np.abs(psi)**2 or np.real).
            y_label (str): Label for the Y axis.
            plot_legend (bool): Whether to show the legend.

        Returns:
            None: Updates the provided axis with the plot.
        """        
        # Index calculation
        indices_to_plot = []
        max_index = len(psi_t) - 1
        
        for t_sec in times_sec:
            index = int(round(t_sec / self.dt))
            indices_to_plot.append(min(index, max_index))
            
        # Sort indeced
        indices_to_plot = sorted(list(set(indices_to_plot)))
        if 0 not in indices_to_plot:
            indices_to_plot.insert(0, 0)
            
        # Colormap
        cmap = cm.viridis
        colors = cmap(np.linspace(0.1, 0.9, len(indices_to_plot)))

        # Plotting loop
        for i, idx in enumerate(indices_to_plot):
            time_sec = idx * self.dt
            color = colors[i]
            
            # Apply transformation (e.g. squared modulus, real, imag)
            y_data = transform_func(psi_t[idx, :])
            
            ax.plot(self.x, y_data, color=color, 
                    label=f"t = {time_sec:.2f}", linewidth=2.0)
            
        # Axis Formatting
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(y_label)
        if plot_legend:
            ax.legend(loc="best", frameon=False, ncols=2)


    def plot_time_velocity(self, T_grid, v, filename="time_velocity.pdf"):
        """
        Plots the average velocity over time and saves it to a file.

        Args:
            T_grid (numpy.ndarray): Array of time points.
            v (numpy.ndarray): Array of velocity values.
            filename (str, optional): Output filename. Defaults to "time_velocity.pdf".

        Returns:
            None: Saves the plot to disk.
        """
        _, ax = plt.subplots()

        ax.plot(T_grid, v, color="black", linewidth=2, linestyle="solid", label=r'$\langle v \rangle(t)$')
        ax.axvline(0, linewidth=1.2, c="purple", linestyle="dashed", label=r"$t=0$")
        ax.axvline(2*np.pi, linewidth=1.2, c="gold", linestyle="dashed", label=r"$t=2\pi$")

        ax.set_title(r"Average velocity in time")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\langle v \rangle$")
        ax.legend(frameon=False, loc="best")

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_spacetime(self, psi_t, T_grid, exp_value, xlim=(0, 10), ylim=(-2, 2.5), filename="heatmap.pdf"):
        """
        Creates a spacetime heatmap plot (x vs t) of the probability density.

        Args:
            psi_t (numpy.ndarray): Wavefunction time evolution array.
            T_grid (numpy.ndarray): Time grid array.
            exp_value (numpy.ndarray): Array of position expectation values.
            xlim (tuple, optional): Limits for the X axis (time). Defaults to (0, 10).
            ylim (tuple, optional): Limits for the Y axis (position). Defaults to (-2, 2.5).
            filename (str, optional): Output filename. Defaults to "heatmap.pdf".

        Returns:
            None: Saves and displays the plot.
        """
        # Compute pdf
        probability_density = np.abs(psi_t)**2

        # Plot the heatmatp
        fig, ax = plt.subplots()

        im = ax.imshow(
            probability_density.T,
            aspect="auto", 
            origin="lower",
            extent=[self.ti, self.simulation_time, self.a, self.b], 
            cmap="viridis",
            vmin=0, 
            vmax=np.max(probability_density)
        )

        ax.plot(T_grid, exp_value, color="white", linewidth=2, label=r"$\langle x \rangle(t)$")
        ax.plot(T_grid, T_grid/self.tf, color="red", linestyle="dashed", linewidth=1, label=r"$q_0(t) = t/t_f$")

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.04)
        cbar.set_label(r"$|\psi(x, t)|^2$")

        ax.legend(loc="best", frameon=True, facecolor="lightgray", framealpha=0.7)

        fig.tight_layout()
        plt.savefig(filename)
        plt.show()

    def time_evo_animation(self, psi_t, filename="wavefunction_evolution.gif", fps=30, show_prob=True):
        """
        Creates and saves an animation of the wavefunction time evolution.

        Args:
            psi_t (numpy.ndarray): Array of wavefunction states over time (N_t, N_x).
            filename (str): Name of the GIF file to save.
            fps (int): Frames per second for the animation.
            show_prob (bool): Whether to show probability density (blue) and real/imag parts (red/green).
        
        Returns:
            None: Saves the animation to disk.
        """        
        # Determine Y limits based on max probability density
        y_max = np.max(np.abs(psi_t)) 
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(self.a, self.b)
        ax.set_ylim(-1.2 * y_max, 1.2 * y_max)
        
        # Initialize lines
        line_prob, = ax.plot([], [], label=r"$|\psi|^2$", color="blue", lw=2.5)
        line_re,   = ax.plot([], [], label=r"$Re(\psi)$", color="red", lw=1.5)
        line_im,   = ax.plot([], [], label=r"$Im(\psi)$", color="green", lw=1.5)
        
        ax.legend(loc="upper right")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\psi(x)$ or $|\psi(x)|^2$")
        ax.set_title("Wavefunction Time Evolution")
        
        # Initialization Function
        def init():
            line_prob.set_data([], [])
            line_re.set_data([], [])
            line_im.set_data([], [])
            return line_prob, line_re, line_im

        # Update Function
        def update(frame):
            # frame is the time index
            psi = psi_t[frame]

            # Data update
            line_prob.set_data(self.x, np.abs(psi)**2)
            line_re.set_data(self.x, np.real(psi))
            line_im.set_data(self.x, np.imag(psi))

            # Title update (time)
            ax.set_title(f"t = {frame * self.dt:.3f} s")

            # Track mean position (packet center)
            if frame > 0:
                 exp_x = np.sum(np.abs(psi)**2 * self.x) * self.dx
                 ax.axvline(exp_x, color='gray', linestyle='--', linewidth=1)

            return line_prob, line_re, line_im

        anim = FuncAnimation(fig, update, frames=len(psi_t), init_func=init, blit=True)

        print(f"Saving the animation '{filename}'...")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print("Saving completed")

        plt.close(fig)