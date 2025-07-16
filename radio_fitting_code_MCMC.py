'''
Code to fit radio light curves using MCMC methods. See Finzell et al. 2018 for details on the
values being fit. See Hjellming et al. (1979) and Wright and Barlow (1975) for details on the model.
'''
import pymc
import numpy as np
from pymc import MCMC
from pylab import hist, show
from pymc.Matplot import plot
import sys
import string
import math as m 
from scipy import optimize
from scipy import integrate
from scipy.special import hyp2f1


##########CONSTANTS##########
k_b = 1.38065e-16 # Boltzmann constant in cgs
m_p = 1.6726219e-24 # Proton mass in grams 
c = 2.997924e10 # Speed of light in cm/s
h = 6.626068e-27 # Planck constant in cgs
const1 = h / k_b

def read_col(file_name, col_number, float_flag=True, skip_line=0):
    col_values = []
    f0 = open(file_name, "r")
    count = 1
    while 1:
        line = f0.readline()
        if not line:
            break
        else:
            if (count > skip_line) and ((len(line.split("#")) == 1)):
                cols = string.split(line)
                if (len(cols) > col_number):
                    try:
                        if float_flag: col_values.append(float(cols[col_number]))
                        else: col_values.append(cols[col_number])
                    except ValueError:
                        pass
            count = count + 1
    f0.close()
    return col_values


def parse_parameter_file(filename, converters):
    line = ''
    with open(filename, 'r') as infile:
        while True:
            while not line.startswith('****'): 
                line = next(infile)  # raises StopIteration, ending the generator
                continue  # find next entry

            entry = {}
            for line in infile:
                line = line.strip()
                if not line: break
                new_line = line.split('#', 1)
                line = new_line[0]
                key, value = map(str.strip, line.split(':', 1))
                if value == "False": value = ""
                if (len(str(value).split(",")) > 1): entry[key] = map(converters.get(key, lambda v: v), str(value).split(","))
                else: entry[key] = converters.get(key, lambda v: v)(value)
            return entry

def fn_func(w_inner, alpha):
    if alpha != 3.0:
        return (3.0 - alpha) / (4.0 * m.pi * (1.0 - m.pow(w_inner, 3.0 - alpha)))
    else:
        return -1.0 / (4.0 * m.pi * m.log(w_inner))

def eta_func(mej, mu, fn, vej, t, alpha, nu, te):
    """
    Computes the composite variable Eta for the model. 

    Parameters
    ----------
    mej : float
        Ejected mass (g).
    mu : float
        Mean molecular weight (g).
    fn : float
        Function of w_inner and alpha.
    vej : float
        Ejecta velocity (cm/s).
    t : float
        Time since explosion (s).
    alpha : float
        Power-law index.
    nu : float
        Frequency (Hz).
    te : float
        Electron temperature (K).

    Returns
    -------
    float
        Eta parameter.
    """
    return (mej * fn / mu) ** 2 * chi_of_nu(nu, te) * (vej * t) ** -5.0

def b_nu_func(te, nu):
    """
    Planck function for blackbody emission at frequency nu and temperature te.

    Parameters
    ----------
    te : float
        Electron temperature (K).
    nu : float
        Frequency (Hz).

    Returns
    -------
    float
        Blackbody intensity (cgs units).
    """
    return 1.4745e-47 * nu ** 3 / (m.exp(4.79924e-11 * (nu / te)) - 1.0)

def gaunt_factor(nu, te):
    """
    Computes the Gaunt factor for free-free emission.

    Parameters
    ----------
    nu : float
        Frequency (Hz).
    te : float
        Electron temperature (K).

    Returns
    -------
    float
        Gaunt factor.
    """
    return (m.sqrt(3.0) / m.pi) * (17.7 + m.log(te ** 1.5 / nu))

def chi_of_nu(nu, te):
    """
    Computes the frequency-dependent opacity.

    Parameters
    ----------
    nu : float
        Frequency (Hz).
    te : float
        Electron temperature (K).

    Returns
    -------
    float
        Opacity.
    """
    return 3.692e8 * (1.0 - m.exp(-const1 * nu / te)) * te ** -0.5 * nu ** -3.0 * gaunt_factor(nu, te)

def generic_tau_integrand(l, a, alpha):
    """
    Integrand for the optical depth calculation.

    Parameters
    ----------
    l : float
        Integration variable.
    a : float
        Impact parameter.
    alpha : float
        Power-law index.

    Returns
    -------
    float
        Value of the integrand.
    """
    return 1.0 / ((a ** 2 + l ** 2) ** alpha)

def generic_tau_func(a, limit1, limit2, index):
    """
    Computes the optical depth integral for given limits.

    Parameters
    ----------
    a : float
        Impact parameter.
    limit1 : float
        Lower integration limit.
    limit2 : float
        Upper integration limit.
    index : float
        Power-law index.

    Returns
    -------
    float
        Integrated optical depth.
    """
    return integrate.quad(lambda l: generic_tau_integrand(l, a, index), limit1, limit2, epsrel=5.0e-1, epsabs=5.0e-1)[0]

def tau1_func_hard(a, eta, w_inner, alpha):
    """
    Optical depth for the 'hard' component, first region.

    Returns
    -------
    float
        Optical depth.
    """
    return eta * abs(generic_tau_func(a, m.sqrt(w_inner ** 2 - a ** 2), m.sqrt(1.0 - a ** 2), alpha))

def tau2_func_hard(a, eta, w_inner, alpha):
    """
    Optical depth for the 'hard' component, second region.

    Returns
    -------
    float
        Optical depth.
    """
    return 2.0 * eta * generic_tau_func(a, 0.0, m.sqrt(1.0 - a ** 2), alpha)

def capital_tau_func(l, a, index):
    """
    Analytical solution for the optical depth integral using hypergeometric function.

    Returns
    -------
    float
        Optical depth.
    """
    return l * a ** (-2.0 * index) * hyp2f1(0.5, index, 1.5, - (l / a) ** 2)

def tau1_func_simple(a, eta, w_inner, alpha):
    """
    Optical depth for the 'simple' component, first region.

    Returns
    -------
    float
        Optical depth.
    """
    return eta * abs(
        capital_tau_func(m.sqrt(1.0 - a ** 2), a, alpha) -
        capital_tau_func(m.sqrt(w_inner ** 2 - a ** 2), a, alpha)
    )

def tau2_func_simple(a, eta, w_inner, alpha):
    """
    Optical depth for the 'simple' component, second region.

    Returns
    -------
    float
        Optical depth.
    """
    return 2.0 * eta * capital_tau_func(m.sqrt(1.0 - a ** 2), a, alpha)

def generic_integrand(a, eta, w_inner, alpha, specific_tau_func):
    """
    Integrand for the flux integral.

    Returns
    -------
    float
        Value of the integrand.
    """
    return a * (1.0 - m.exp(-2.0 * specific_tau_func(a, eta, w_inner, alpha)))

def component_integral(eta, w_inner, alpha, region, tau_func):
    """
    Computes the integral for a given region and tau function.

    Parameters
    ----------
    eta : float
        Eta parameter.
    w_inner : float
        Inner radius ratio.
    alpha : float
        Power-law index.
    region : tuple
        (lower, upper) integration limits.
    tau_func : function
        Function to compute tau.

    Returns
    -------
    float
        Integrated value.
    """
    lower, upper = region
    return integrate.quad(
        lambda a: generic_integrand(a, eta, w_inner, alpha, tau_func),
        lower, upper, epsrel=5.0e-1, epsabs=5.0e-1
    )[0]

def actual_integrals_hard(eta, w_inner, alpha):
    """
    Computes the total integral for the 'hard' case.

    Returns
    -------
    float
        Total integrated value.
    """
    first = component_integral(eta, w_inner, alpha, (0.0, w_inner), tau1_func_hard)
    second = component_integral(eta, w_inner, alpha, (w_inner, 1.0), tau2_func_hard)
    return first + second

def actual_integrals_simple(eta, w_inner, alpha):
    """
    Computes the total integral for the 'simple' case.

    Returns
    -------
    float
        Total integrated value.
    """
    first = component_integral(eta, w_inner, alpha, (0.0, w_inner), tau1_func_simple)
    second = component_integral(eta, w_inner, alpha, (w_inner, 1.0), tau2_func_simple)
    return first + second

def s_sub_nu(t, w_inner, vej, mej, mu, alpha, nu, te, d):
    """
    Computes the model flux at frequency nu and time t. Note that the difference between simple 
    and hard cases is determined by the value of alpha.

    Parameters
    ----------
    t : float
        Time since explosion (s).
    w_inner : float
        Inner radius ratio.
    vej : float
        Ejecta velocity (cm/s).
    mej : float
        Ejected mass (g).
    mu : float
        Mean molecular weight (g).
    alpha : float
        Power-law index.
    nu : float
        Frequency (Hz).
    te : float
        Electron temperature (K).
    d : float
        Distance (cm).

    Returns
    -------
    float
        Model flux (cgs units).
    """
    fn = fn_func(w_inner, alpha)
    eta = eta_func(mej, mu, fn, vej, t, alpha, nu, te)
    prefactor = 2.0 * m.pi * b_nu_func(te, nu) * (vej * t) ** 2 / d ** 2
    if alpha < 5.0:
        integral = actual_integrals_simple(eta, w_inner, alpha)
    else:
        integral = actual_integrals_hard(eta, w_inner, alpha)
    return prefactor * integral


alpha = 2.0
mu = 2.33e-24
te = 1.0e4
def model(t, nu, observed_flux_values, observed_flux_value_errors): 
    """
    Defines a probabilistic model for fitting observed radio flux values using MCMC.

    Parameters
    ----------
    t : array-like
        Array of time values corresponding to the observations.
    nu : array-like
        Array of frequency values for each observation.
    observed_flux_values : array-like
        Observed flux values to be fitted by the model.
    observed_flux_value_errors : array-like
        Errors associated with each observed flux value.

    Returns
    -------
    dict
        A dictionary of local variables including the PyMC stochastic and deterministic variables
        that define the model.

    Notes
    -----
    - The model defines four uniform priors: `w_inner_val`, `vej_val`, `log_mej_val`, and `d_val`.
    - The deterministic function `my_func` computes the expected flux values using the `s_sub_nu` function,
      which models the physical process given the parameters.
    - The observed data is modeled as a normal distribution centered on the deterministic prediction,
      with precision determined by the observed errors.
    - This function is intended for use with PyMC for Bayesian inference and MCMC sampling.
    """
    w_inner_val = pymc.Uniform('w_inner_val', lower=0.1, upper=0.9, doc='Inner Ratio Point')
    vej_val = pymc.Uniform('vej_val', lower=1000.0, upper=6000.0, doc='Ejecta Velocity')
    log_mej_val = pymc.Uniform('log_mej_val', lower=-5, upper=-2, doc='Log of Mass Ejected')
    d_val = pymc.Uniform('d_val', lower=1.0, upper=20.0, doc='Distance')

    @pymc.deterministic(plot=False)
    def my_func(w_inner_val=w_inner_val, vej_val=vej_val, log_mej_val=log_mej_val, d_val=d_val):
        all_s_sub_nu_values = np.array([
            s_sub_nu(
                t[hh], w_inner_val, vej_val * 1.0e5, m.pow(10.0, log_mej_val) * 1.981e33,
                mu, alpha, nu[hh], te, d_val * 3.0856e21
            ) for hh in range(t.shape[0])
        ])
        return all_s_sub_nu_values
    y = pymc.Normal('y', mu=my_func, tau=1.0 / np.power(observed_flux_value_errors, 2.0), value=observed_flux_values, observed=True)
    return locals()


# converters = {'FILENAME': str,
#               'PLOTFILENAME': str,
#               'MASS': float,
#               'TEMPERATURE': float,
#               'VELOCITY': float,
#               'WINNER': float,
#               'DISTANCE': float,
#               'ALPHA': float}


def main():
    if (len(sys.argv) < 2):
        print("USAGE: python radio_MCMC_fitting_code.py ParameterFile.txt")
        sys.exit()
    # param_file_name = sys.argv[1]
    # parameters = parse_parameter_file(param_file_name, converters)
    filename = sys.argv[1]
    t_julian_values = np.array(read_col(filename, 0)) * 24.0 * 60.0 * 60.0
    nu_values = np.array(read_col(filename, 1)) * 1.0e9
    s_values = np.array(read_col(filename, 2)) * 1.0e-26
    s_error_values = np.array(read_col(filename, 3)) * 1.0e-26

    M = MCMC(model(t_julian_values, nu_values, s_values, s_error_values), db='pickle', dbname='RadioFits_AlphaOfTwo2.pickle')

    M.sample(iter=10000, burn=2500)
    M.db.close()
