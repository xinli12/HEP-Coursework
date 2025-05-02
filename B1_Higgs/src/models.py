import numpy as np
from numba_stats import norm, crystalball, bernstein


# -----------------------
# Signal Models
# -----------------------

def cbg_pdf(x, mu, f, sg1, sg2, b, m):
    """
    Composite signal model: Gaussian + Crystal Ball PDF.
    
    Parameters:
        x (array): Input data.
        mu (float): Mean of Gaussian.
        f (float): Gaussian fraction.
        sg1 (float): Sigma of Gaussian.
        sg2 (float): Sigma of Crystal Ball.
        b (float): Crystal Ball 'beta' parameter.
        m (float): Crystal Ball 'm' parameter.
        
    Returns:
        array: Probability density.
    """
    return f * norm.pdf(x, mu, sg1) + (1 - f) * crystalball.pdf(x, b, m, mu, sg2)


def cbg_cdf(x, mu, f, sg1, sg2, b, m):
    """
    Composite signal model: Gaussian + Crystal Ball CDF.
    """
    return f * norm.cdf(x, mu, sg1) + (1 - f) * crystalball.cdf(x, b, m, mu, sg2)


# -----------------------
# Background Model
# -----------------------

def bernstein_pdf(x, b0, b1, b2, b3, b4, xmin=100, xmax=160):
    """
    Background PDF using a 4th-order Bernstein polynomial.
    
    Normalised over [xmin, xmax].
    """
    coeffs = [b0, b1, b2, b3, b4]
    pdf = bernstein.density(x, coeffs, xmin, xmax)
    norm_factor = np.diff(bernstein.integral([xmin, xmax], coeffs, xmin, xmax))
    return pdf / norm_factor


def bernstein_cdf(x, b0, b1, b2, b3, b4, xmin=100, xmax=160):
    """
    Background CDF using a 4th-order Bernstein polynomial.
    """
    coeffs = [b0, b1, b2, b3, b4]
    cdf = bernstein.integral(x, coeffs, xmin, xmax)
    norm_factor = np.diff(bernstein.integral([xmin, xmax], coeffs, xmin, xmax))
    return cdf / norm_factor


