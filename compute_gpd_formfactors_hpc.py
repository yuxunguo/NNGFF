#!/usr/bin/env python3
"""
HPC version: Compute GPD form factors (AGFF, DGFF, HCFF, ECFF) for a square lattice of (ξ, t) values.

This script is designed for HPC job submission where H_alpha and H_beta are treated as variables
and all other parameters are declared as constants.

Usage:
    python compute_gpd_formfactors_hpc.py <H_alpha> <H_beta> [output_file]

This script generates:
- 50 points for t in [-10, 0] GeV²
- 50 points for ξ in [0.3, 0.8]
- Total: 2500 (ξ, t) pairs

For each pair, it computes:
- AGFF: Axial gluon form factor
- DGFF: D-term gluon form factor  
- HCFF: Hard Compton form factor (H)
- ECFF: Hard Compton form factor (E)
"""

import numpy as np
import pandas as pd
import csv
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
from scipy.integrate import IntegrationWarning

# Suppress all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=IntegrationWarning)

# Add parent directory to path 
# to import GPDgen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPDgen import HCFF, ECFF, HGFF

# Constants - these parameters are fixed for HPC runs
CONST_PARAMS = {
    'H_norm': 0.5,      # Normalization (Ag0)
    'H_m': 1.5,         # Mass scale (MAg)
    'D1': -0.5,         # D-term coefficient (Dg0)
    'mD1': 2.0,         # D-term mass scale (MDg)
    'include_n3': False # Whether to include n=3 D-term
}

def compute_formfactors(xi, t, **params):
    """
    Compute all GPD form factors for given (xi, t) pair.
    
    Parameters:
    -----------
    xi : float
        Skewness parameter
    t : float  
        Momentum transfer squared (GeV²)
    **params : dict
        GPD model parameters
        
    Returns:
    --------
    dict : Dictionary containing AGFF, DGFF, HCFF_real, HCFF_imag, ECFF_real, ECFF_imag
    """
    
    # Compute HGFF (AGFF, DGFF)
    AGFF, DGFF = HGFF(xi, t, **params)
    
    # Compute HCFF and ECFF
    HCFF_complex = HCFF(xi, t, **params)
    ECFF_complex = ECFF(xi, t, **params)
    
    return {
        'AGFF': AGFF,
        'DGFF': DGFF,
        'HCFF_real': HCFF_complex.real,
        'HCFF_imag': HCFF_complex.imag,
        'ECFF_real': ECFF_complex.real,
        'ECFF_imag': ECFF_complex.imag
    }

def worker_function(xi_t_pair, params):
    """
    Worker function for multiprocessing.
    
    Parameters:
    -----------
    xi_t_pair : tuple
        (xi, t) pair
    params : dict
        GPD model parameters
        
    Returns:
    --------
    dict : Result dictionary with xi, t, and form factors
    """
    xi, t = xi_t_pair
    
    try:
        # Compute form factors
        formfactors = compute_formfactors(xi, t, **params)
        
        # Return result
        return {
            'xi': xi,
            't': t,
            'AGFF': formfactors['AGFF'],
            'DGFF': formfactors['DGFF'],
            'HCFF_real': formfactors['HCFF_real'],
            'HCFF_imag': formfactors['HCFF_imag'],
            'ECFF_real': formfactors['ECFF_real'],
            'ECFF_imag': formfactors['ECFF_imag']
        }
        
    except Exception as e:
        # Return NaN values for failed computations (silently)
        return {
            'xi': xi,
            't': t,
            'AGFF': np.nan,
            'DGFF': np.nan,
            'HCFF_real': np.nan,
            'HCFF_imag': np.nan,
            'ECFF_real': np.nan,
            'ECFF_imag': np.nan
        }

def generate_lattice_data(H_alpha, H_beta, output_file="gpd_formfactors.csv"):
    """
    Generate GPD form factors for a square lattice of (ξ, t) values using multiprocessing.
    
    Parameters:
    -----------
    H_alpha : float
        Forward distribution parameter (variable)
    H_beta : float
        Forward distribution parameter (variable)
    output_file : str
        Output CSV filename
    """
    
    # Combine variable parameters with constants
    params = {
        'H_alpha': H_alpha,
        'H_beta': H_beta,
        **CONST_PARAMS
    }
    
    # Define the lattice
    t_values = np.linspace(-10.0, 0.0, 50)  # 50 points from -10 to 0 GeV²
    xi_values = np.linspace(0.3, 0.8, 50)  # 50 points from 0.1 to 0.8
    
    # Get number of CPU cores
    n_cores = cpu_count()
    
    # Create all (xi, t) pairs
    xi_t_pairs = [(xi, t) for t in t_values for xi in xi_values]
    total_combinations = len(xi_t_pairs)
    
    # Create partial function with fixed params
    worker_func = partial(worker_function, params=params)
    
    # Use multiprocessing without progress bar for HPC
    with Pool(processes=n_cores) as pool:
        results = list(pool.map(worker_func, xi_t_pairs))
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    df.to_csv(output_path, index=False)
    
    return df

def main():
    """
    Main function to run the computation with command-line arguments.
    Usage: python compute_gpd_formfactors_hpc.py <H_alpha> <H_beta> [output_file]
    """
    
    # Get command-line arguments
    if len(sys.argv) < 3:
        sys.exit(1)
    
    try:
        H_alpha = float(sys.argv[1])
        H_beta = float(sys.argv[2])
        # Generate filename based on parameters
        output_file = f"alpha_{H_alpha}_beta_{H_beta}.csv"
    except ValueError:
        sys.exit(1)
    
    # Generate the data
    df = generate_lattice_data(H_alpha, H_beta, output_file)
    
    return df

if __name__ == "__main__":
    # Run the computation
    df = main()
