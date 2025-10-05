import numpy as np
from iminuit import Minuit
from scipy.integrate import quad, IntegrationWarning
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import csv
from GPDgen import HCFF, ECFF, HGFF
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import shutil

warnings.filterwarnings("ignore", category=IntegrationWarning)

NF=4

Mproton = 0.938
MJpsi = 3.097
Mcharm = MJpsi/2
alphaEM = 1/133
alphaS = 0.30187
psi2 = 1.0952 /(4 * np.pi)

conv = 2.5682 * 10 ** (-6)

def Kpsi(W :float):
    return np.sqrt(((W**2 -(MJpsi - Mproton)**2) *( W**2 -(MJpsi + Mproton)**2 ))/(4.*W**2))

def PCM(W: float):
    return np.sqrt(( W**2 - Mproton**2 )**2/(4.*W**2))

def tmin(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) - 2 * Kpsi(W) * PCM(W)

def tmax(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) + 2 * Kpsi(W) * PCM(W)

def PPlus(W: float):
    return W/np.sqrt(2)

def PprimePlus(W: float, t: float):
    return np.sqrt(Mproton**2 + Kpsi(W)**2)/np.sqrt(2) + (-2*Mproton**2 + t + 2*np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)))/(2.*np.sqrt(2)*PCM(W))

def PbarPlus2(W: float, t: float):
    return ( PPlus(W) + PprimePlus(W,t) ) ** 2 / 4

def DeltaPlus2(W: float, t: float):
    return (PprimePlus(W,t) - PPlus(W) ) ** 2

def Xi(W: float, t: float):
    return (PPlus(W) - PprimePlus(W,t))/(PPlus(W) + PprimePlus(W,t))

def WEb(Eb: float):
    return np.sqrt(Mproton)*np.sqrt(Mproton + 2 * Eb)


def dsigma_prefact(W: float):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi) ** 2/ (3 * MJpsi ** 3) * psi2

def G2(W: float, t: float, **kwargs):
    
    prefact = dsigma_prefact(W)
    
    xi = Xi(W ,t)
    HC =  HCFF(xi,t, **kwargs)
    EC =  ECFF(xi,t, **kwargs)
    
    return prefact*((1-xi**2) * HC.conjugate()*HC 
            - 2* xi**2 * (HC.conjugate()*EC)
            -(xi**2 + t/(4* Mproton**2)) * EC.conjugate()*EC).real

'''
def generate_xsec(EbMin, EbMax, params, n=2, dt=1.0):
    Es = np.linspace(EbMin, EbMax, n)   # n points for W
    results = []
    for E in Es:
        W = WEb(E)
        t_start, t_end = tmin(W), tmax(W)
        
        t_start = max(t_start, -10.0)

        num_points = int(round((t_end - t_start) / dt)) + 1
        ts = np.linspace(t_start, t_end, num_points)
        
        for t in ts:
            G2WT = G2(W,t, **params).real
            results.append((E, t, G2WT))  # (W, t, F(W,t))
            
    return results
'''

def generate_xsec(xsecpairs, params):
    
    results = []
    for E, t in xsecpairs:
        W = WEb(E)
        G2WT = G2(W, t, **params).real
        results.append((E, t, G2WT))  # (W, t, F(W,t))
    return results

def generate_GFF(dtGFF, params):
    t_start = -10.0
    t_end   = 0.
    num_points = int(round((t_end - t_start) / dtGFF)) + 1
    ts = np.linspace(t_start, t_end, num_points)
    results = []
    
    for t in ts:
        AGFF, DGFF = HGFF(1.0, t, **params)
        results.append((t, AGFF, DGFF)) 
    
    return results

def process_params(arg_tuple):
    
    i, params, extra_args, total_files = arg_tuple
    
    xsecpairs = extra_args.get("xsecpairs")
    dtGFF = extra_args.get("dtGFF")
    output_dir = extra_args.get("output_dir")

    pairs_xsec = generate_xsec(xsecpairs, params= params)    
    pairs_GFF = generate_GFF(dtGFF=dtGFF, params= params)
    # Compute number of digits for leading zeros
    num_digits = len(str(total_files))
    filename_xsec= f"{output_dir}/xsec/sample{str(i).zfill(num_digits)}.csv"

    os.makedirs(f"{output_dir}/xsec", exist_ok=True)
    os.makedirs(f"{output_dir}/GFF", exist_ok=True) 
        
    with open(filename_xsec, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["E", "t", "xsec"])
        writer.writerows(pairs_xsec)
    
    filename_GFF = f"{output_dir}/GFF/sample{str(i).zfill(num_digits)}.csv"
    
    with open(filename_GFF, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "AGFF","DGFF"])
        writer.writerows(pairs_GFF)
        
    return filename_xsec, len(pairs_xsec)

if __name__ == "__main__":
    
    dsigmadata = pd.read_csv("Diff_xsec_combined.csv")
    pairs = list(zip(dsigmadata["avg_E"], -dsigmadata["avg_abs_t"]))

    # Example parameter ranges
    Output_Dir = "Output/DoubleDist"
    extra_args = {"xsecpairs": pairs, "dtGFF": 0.2, "output_dir": f"{Output_Dir}"}
    
    print(len(pairs))

    param_ranges = {
        "H_alpha": {"start": -1.0, "end": -0.25, "step": 0.25},
        "H_beta": {"start": 1.0, "end": 7.0, "step": 2.0},
        "H_norm": {"start": 0.25, "end": 1.0, "step": 0.25},  # Ag0
        "H_m": {"start": 0.5, "end": 2.5, "step": 1.0},     # MAg
        "D1": {"start": -1.0, "end": 1.0, "step": 1.0},     # Dg0
        "mD1": {"start": 0.5, "end": 2.5, "step": 1.0}     # MDg
    }
    
    param_values = {
        k: np.linspace(
            v["start"],
            v["end"],
            int(round((v["end"] - v["start"]) / v["step"])) + 1
        )
        for k, v in param_ranges.items()
    }

    # Generate all combinations
    keys = list(param_values.keys())
    all_combinations = product(*(param_values[k] for k in keys))

    # Convert to list of dictionaries
    dict_list = [dict(zip(keys, values)) for values in all_combinations]

    # Example: print first 5
    for d in dict_list[:5]:
        print(d)
    
    total_files = len(dict_list)
    
    print(total_files)
    
    args_list = [(i, params, extra_args, total_files) for i, params in enumerate(dict_list, start=1)]
    #'''
    if os.path.exists(Output_Dir):
        shutil.rmtree(Output_Dir)
    
    os.makedirs(Output_Dir, exist_ok=True)
    
    with Pool() as pool:
        # Use imap_unordered so we can update progress as tasks complete
        results = []
        for result in tqdm(pool.imap_unordered(process_params, args_list), total=len(args_list)):
            results.append(result)

    _, nrows = results[0]

    print(f"Saved in {Output_Dir} with {total_files} sets of parameters and {nrows} of cross-sections each set")
    #'''
