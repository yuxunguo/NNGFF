import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import gegenbauer
from scipy.special import beta as BETA

# ----- D-term with optional n=3 -----
def D_g(x, xi, t, **kwargs):
    """
    Gluon D-term:
    - n=1 term is always included: D1 * (1 - z^2)^2 / (1 - t/mD1^2)^3
    - n=3 term is optional: D3 * C2^{5/2}(z) / (1 - t/mD3^2)^3
    z = x / xi, |x| < |xi|
    """
    
    D1=kwargs.get("D1")
    mD1=kwargs.get("mD1")
    include_n3=kwargs.get("include_n3", False)
    D3=kwargs.get("D3", 0.0)
    mD3=kwargs.get("mD3", 1.0)
    
    if abs(x) >= abs(xi):
        return 0.0  # outside ERBL region

    z = x / xi
    # n=1 term (C0^{5/2} = 1)
    tripole1 = 1 / (1 - t/mD1**2)**3
    Dval = D1 * (1 - z**2)**2 * tripole1

    # n=3 term (optional)
    if include_n3 and D3 != 0.0:
        tripole3 = 1 / (1 - t/mD3**2)**3
        C2 = gegenbauer(2, 2.5)(z)  # C_{2}^{5/2}(z)
        Dval += D3 * (1 - z**2)**2 * C2 * tripole3

    return Dval*3/2*5/4


# ----- Profile function -----
def pi_g(alpha, beta_abs):
    num = ((1 - beta_abs)**2 - alpha**2)**2
    denom = (1 - beta_abs)**5
    return (15/16) * num / denom if denom > 0 else 0.0

# ----- Forward distribution -----
def f_g(beta_abs, t, **kwargs):
    """
    Forward gluon distribution:
    norm * (beta_abs**alpha) * ((1 - beta_abs)**beta) / (1 - t/m**2)^3
    """
    alpha = kwargs.get("alpha")
    beta = kwargs.get("beta")
    norm  = kwargs.get("norm")
    m     = kwargs.get("m")

    denom = (1 - t/m**2)**3
    
    if beta_abs == 0.0 and alpha <= 0:  # regulator
        return 0.0
    
    return norm/BETA(2+alpha,1+beta) * (beta_abs**alpha) * ((1 - beta_abs)**beta) / denom

# ----- H_g^{DD} -----
def H_g_DD(x, xi, t, **kwargs):
    if abs(xi) < 1e-12:
        beta_abs = abs(x)
        alpha_min, alpha_max = -1 + beta_abs, 1 - beta_abs
        integral, _ = quad(lambda alpha: pi_g(alpha, beta_abs), alpha_min, alpha_max)
        return beta_abs * f_g(beta_abs, t, **kwargs) * integral

    def integrand(beta):
        beta_abs = abs(beta)
        alpha = (x - beta) / xi
        if abs(alpha) > 1 - beta_abs:
            return 0.0
        return (beta_abs / abs(xi)) * pi_g(alpha, beta_abs) * f_g(beta_abs, t, **kwargs)

    result, _ = quad(integrand, -1, 1)
    return result

def H_g(x, xi, t, **kwargs):
    """
    Full gluon GPD: H_g = H_g^{DD} + |xi| * D_g(x, xi, t)
    kwargs passed to H_g_DD and D_g (D1, mD1, include_n3, D3, mD3)
    """
    
    HDD = H_g_DD(x, xi, t, alpha = kwargs.get("H_alpha"),
                           beta  = kwargs.get("H_beta"),
                           norm  = kwargs.get("H_norm"),
                           m     = kwargs.get("H_m"))

    Dterm = abs(xi) * D_g(x, xi, t,
                           D1=kwargs.get("D1"),
                           mD1=kwargs.get("mD1"),
                           include_n3=kwargs.get("include_n3", False),
                           D3=kwargs.get("D3"),
                           mD3=kwargs.get("mD3")) \
            if abs(x) < abs(xi) else 0.0
    return HDD + Dterm

def E_g(x, xi, t, **kwargs):
    """
    Full gluon GPD: H_g = H_g^{DD} + |xi| * D_g(x, xi, t)
    kwargs passed to H_g_DD and D_g (D1, mD1, include_n3, D3, mD3)
    """
    Dterm = abs(xi) * D_g(x, xi, t,
                           D1=kwargs.get("D1"),
                           mD1=kwargs.get("mD1"),
                           include_n3=kwargs.get("include_n3", False),
                           D3=kwargs.get("D3"),
                           mD3=kwargs.get("mD3")) \
            if abs(x) < abs(xi) else 0.0
    return - Dterm


# ----- Convolution integral -----
def HCFF(xi, t, **kwargs):
    def integrand(x):
        if abs(x - xi) < 1e-8 or abs(x + xi) < 1e-8:
            return 0.0
        return H_g(x, xi, t, **kwargs)* (1/ (x + xi)-1/(x- xi)) /(2* xi)

    real_part, _ = quad(integrand, -1, 1, limit=200)
    imag_part = np.pi/xi * H_g(xi, xi, t, **kwargs)
    return real_part + 1j * imag_part

def ECFF(xi, t, **kwargs):
    def integrand(x):
        if abs(x - xi) < 1e-8 or abs(x + xi) < 1e-8:
            return 0.0
        return E_g(x, xi, t, **kwargs)* (1/ (x + xi)-1/(x- xi)) /(2* xi)

    real_part, _ = quad(integrand, -1, 1, limit=200)
    imag_part = np.pi/xi * E_g(xi, xi, t, **kwargs)
    return real_part + 1j * imag_part

def HGFF(xi, t, **kwargs):

    def integrand(x):
        return H_g_DD(x, xi, t, alpha = kwargs.get("H_alpha"),
                           beta  = kwargs.get("H_beta"),
                           norm  = kwargs.get("H_norm"),
                           m     = kwargs.get("H_m"))
        

    def integrandD(x):
        return abs(xi) * D_g(x, xi, t,
                           D1=kwargs.get("D1"),
                           mD1=kwargs.get("mD1"),
                           include_n3=kwargs.get("include_n3", False),
                           D3=kwargs.get("D3"),
                           mD3=kwargs.get("mD3")) \
                if abs(x) < abs(xi) else 0.0

    AGFF, _ = quad(integrand, 0, 1, limit=200)
    DGFF, _ = quad(integrandD, 0, 1, limit=200)

    return AGFF, DGFF/xi**2



# ----- Example usage -----
if __name__ == "__main__":
    
    xi, t = 0.2, -1.0
    Ag0 = 0.4
    MAg = 1.6
    Dg0 = -1.0
    MDg = 1.0
    HCFF = HCFF(xi, t, H_alpha=-0.2, H_beta=6.0, H_norm=Ag0, H_m=MAg, D1 = Dg0, mD1 = MDg)
    ECFF = ECFF(xi, t, H_alpha=-0.2, H_beta=6.0, H_norm=Ag0, H_m=MAg, D1 = Dg0, mD1 = MDg)
    AGFF, DGFF = HGFF(xi, t, H_alpha=-0.2, H_beta=6.0, H_norm=Ag0, H_m=MAg, D1 = Dg0, mD1 = MDg)
    
    print(HCFF)
    print(5/4*2*(AGFF+xi**2*DGFF)/xi**2)
    
    print(ECFF)
    print(5/4*2*(-xi**2*DGFF)/xi**2)
    
    print(Ag0/(1-t/MAg**2)**3, AGFF)
    print(Dg0/(1-t/MDg**2)**3, DGFF)

