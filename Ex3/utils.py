import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.linalg import eigsh

from debugger import matrix_checkpoints



def generate_hermitian_matrix(dim, distr="uniform", low=0.0, high=1.0, mean=0.0, std=1.0, complex=False):
    """
    Function that generates hermitian matrices.

    Args:
        dim (int): input dimension of the matrix.
        distr (str, optional): distribution to generate data. Defaults to "uniform".
        low (float, optional): lower bound for uniform distribution. Defaults to 0.0.
        high (float, optional): upper bound for uniform distribution. Defaults to 1.0.
        mean (float, optional): mean of normal distribution. Defaults to 0.0.
        std (float, optional): variance of normal distribution. Defaults to 1.0.
        complex (bool, optional): generate a complex matrix. Defaults to False.

    Raises:
        TypeError: uncorrect distribution inserted

    Returns:
        (float, float): output matrix
    """

    if not complex:
        # Generate the real matrix according to the specified distribution
        if distr == "uniform":
            A = np.random.uniform(low, high, size=dim*dim).reshape(dim, dim)
        elif distr == "normal":
            A = np.random.normal(mean, std, size=dim*dim).reshape(dim,dim)
        else:
            raise TypeError("ERROR: uncorrect distribution inserted!\n" \
                            "Please choose between 'uniform' and 'normal'")
        
        # Make the matrix hermitian
        A = A + A.T

        # Check if it is effectively hermitian
        if matrix_checkpoints(0, A) == True:
            return A
    else:
        # Generate complex matrix according to the specified distribution
        if distr == "uniform":
            Re = np.random.uniform(low, high, size=dim*dim).reshape(dim, dim)
            Im = np.random.uniform(low, high, size=dim*dim).reshape(dim, dim)
        elif distr == "normal":
            Re = np.random.normal(mean, std, size=dim*dim).reshape(dim,dim)
            Im = np.random.normal(mean, std, size=dim*dim).reshape(dim,dim)
        else:
            raise TypeError("ERROR: uncorrect distribution inserted!\n" \
                            "Please choose between 'uniform' and 'normal'")
        
        # Make the matrix complex and hermitian
        A = Re + 1j*Im
        A = A + A.conjugate().T

        # Check if it is effectively hermitian
        if matrix_checkpoints(0, A)==True:
            return A

def generate_diagonal_matrix(dim, distr="uniform", low=0.0, high=1.0, mean=0.0, std=1.0):
    """
    Function that generates diagonal matrices.

    Args:
        dim (int): input dimension of the matrix.
        distr (str, optional): distribution to generate data. Defaults to "uniform".
        low (float, optional): lower bound for uniform distribution. Defaults to 0.0.
        high (float, optional): upper bound for uniform distribution. Defaults to 1.0.
        mean (float, optional): mean of normal distribution. Defaults to 0.0.
        std (float, optional): variance of normal distribution. Defaults to 1.0.
        complex (bool, optional): generate a complex matrix. Defaults to False.

    Raises:
        TypeError: uncorrect distribution inserted

    Returns:
        (float, float): output matrix
    """

    if distr=="uniform":
        B = np.diag(np.random.uniform(low, high, dim))
    elif distr=="normal":
        B = np.diag(np.random.normal(mean, std, dim))
    else: 
        raise TypeError("ERROR: uncorrect distribution inserted!\n" \
                        "Please choose between 'uniform' and 'normal'")
    
    # Check if it is effectively diagonal
    if matrix_checkpoints(1, B)==True:
        return B



def generate_sparse_random_matrix(dim, density=0.01):
    """
    Function that generates sparse random matrices.

    Args:
        dim (int): matrix dimension.
        density (float, optional): density. Defaults to 0.01.

    Returns:
        (float, float): output matrix
    """

    # Generate non-zero elements
    nonzero_el = int(density * dim * dim)

    # Generate random index pairs
    idx = set()
    while len(idx) < nonzero_el:
        i = np.random.randint(0, dim)
        j = np.random.randint(0, dim)
        idx.add((i, j))

    rows, cols = zip(*idx)

    # Generate values
    values = np.random.randn(len(rows))

    # Define the symmetric sparse matrix
    S = coo_matrix((values, (rows, cols)), shape=(dim, dim))
    #S = (S + S.T) * 0.5 # Make it hermitian
    S = (S + S.conj().T) / 2 # Make it hermitian

    return S



def compute_eigenvals_spacing(A, k=None):
    """
    Function that compute the normalized eigenvalue spacing of a given matrix

    Args:
        A (float, float) or (complex, complex): input dimension
        k (int, optional): number of eigenvalues to compute. Defaults to None.

    Returns:
        list: list of normalized eigenvalues spacing.
    """

    if issparse(A):
        dim = A.shape[0]

        if k is None:
            k = dim-1
        
        # Compute the eigenvalues and sort them in ascending order
        eigenval = np.sort(eigsh(A, k=k, which="LM")[0])
    
    else:
        # Compute the eigenvalues and sort them in ascending order
        eigenval = np.sort(scipy.linalg.eigvalsh(A))

    # compute the space between the eigenvalues
    Lambda_i = np.diff(eigenval)
    Lambda_i = Lambda_i[:-1] # discard the largest eigenvalue
    Lambda_mean = np.mean(Lambda_i)

    # normalize space between the eigenvalues
    s = Lambda_i/Lambda_mean

    return s



def P(s, a, b, alpha, beta):
    """
    Function to define the matrix distribution fit.

    Args:
        s (float): fit parameter.
        a (float): fit parameter.
        b (float): fit parameter.
        alpha (float): fit parameter.
        beta (float): fit parameter.

    Returns:
        function: fitting function.
    """
    return a * s**alpha * np.exp(-b * s**beta)



def fit_P(bin_centers, pdf):
    """
    Function to generate the matrix distribution fit.

    Args:
        bin_centers (float): centers of the bin for the distribution.
        pdf (float): probability density function.

    Returns:
        _type_: fit parameters.
    """
    popt, pcov = curve_fit(P, bin_centers, pdf)
    return popt, pcov



def compute_P(all_s, nbins):
    """
    function to compute the eigenvalues distribution.

    Args:
        all_s (list, float): list of all the normalized eigenvalues spacings.
        nbins (int): number of bins.

    Returns:
        float: bin edges.
        int: counts for each bin.
    """
    # Define bin edges
    bin_edges = np.linspace(np.min(all_s), np.max(all_s), nbins+1)
    counts = []

    for i in range(nbins):
        c = 0
        for j in all_s:
            if bin_edges[i] < j <= bin_edges[i+1]:
                c += 1
        counts.append(c)

    return bin_edges, np.array(counts)


def wigner_surmise(s):
    """
    Function to define the Wigner surmise distribution.

    Args:
        s (float): fit parameter.

    Returns:
        function: Wigner surmise function.
    """
    return 32/(np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)



def plot_P(bin_edges, counts, title=None, color="blue", label="Generated pdf", filename=None, fit=True, fit_WS=True):
    """
    Function to plot the normalized eigenvalues spacing.

    Args:
        bin_edges (float): bin edges.
        counts (int): counts for each bin.
        title (str, optional): plot title. Defaults to None.
        color (str, optional): histogram color. Defaults to "blue".
        label (str, optional): plot labels. Defaults to "Generated pdf".
        filename (str, optional): filename for saving the plot. Defaults to None.
        fit (bool, optional): fit or not. Defaults to True.
        fit_WS (bool, optional): Wigner surmise fit or not. Defaults to True.
    """

    widths = np.diff(bin_edges)
    N = np.sum(counts)

    # Define the pdf
    pdf = counts/(N*widths)

    # Plot pdf
    plt.bar(bin_edges[:-1], pdf, width=widths, alpha=0.5, color=color, align="edge", label=label)


    if (fit):
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        popt1, pcov1 = fit_P(bin_centers, pdf)

        # Print fit results
        print(f"a: {popt1[0]} +/- {np.sqrt(pcov1[0, 0])}")
        print(f"b: {popt1[1]} +/- {np.sqrt(pcov1[1, 1])}")
        print(f"alpha: {popt1[2]} +/- {np.sqrt(pcov1[2, 2])}")
        print(f"beta: {popt1[3]} +/- {np.sqrt(pcov1[3, 3])}")

        # Plot fitted curve
        x = np.linspace(bin_edges[0]+0.025, 9.5, 1000)
        label_fit1 = (r"$P(s) = %.2f\, s^{%.2f} \, e^{-%.2f\, s^{%.2f}}$"%(popt1[0], popt1[2], popt1[1], popt1[3]))
        plt.plot(x, P(x, *popt1), '--', linewidth=1.5, label=label_fit1, color="black")

        if (fit_WS):
            label_fit2 = r"Wigner surmise: $P(s) = \frac{32}{\pi^2} s^2 e^{-\frac{4}{\pi} s^2}$"
            plt.plot(x, wigner_surmise(x), '--', linewidth=1.5, label=label_fit2, color="red")

    if(title):
        plt.title(title + " pdf", fontsize=16)

    plt.xlabel("Normalized eigenvalues spacing", fontsize=16)
    plt.ylabel("PDF", fontsize=16)
    plt.xlim(-0.5, 10)
    plt.legend(loc="best", frameon=False, fontsize=16)
    plt.tight_layout()

    if (filename):
        plt.savefig(filename)