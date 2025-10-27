#!/usr/bin/env julia
"""
Main Julia script for GPD form factors computation.

This script computes GPD form factors (AGFF, DGFF, HCFF, ECFF) for a parameter grid
of H_alpha and H_beta values using parallel processing.

Usage:
    julia --threads=N main.jl [--test]
    julia main.jl --threads N [--test]

Parameters:
- H_alpha: 100 values from -1.9 to -0.1
- H_beta: 100 values from 3.0 to 15.0
- Total: 10,000 parameter combinations
- Each combination: 10×10 = 100 (xi, t) points (reduced for speed)
- Total computations: 1,000,000 points

Expected runtime: varies with system performance and thread count
"""

using QuadGK
using SpecialFunctions
using CSV
using DataFrames
using Dates
using Base.Threads

# Constants - these parameters are fixed for all threads
const H_norm = 0.5      # Normalization (Ag0)
const H_m = 1.5         # Mass scale (MAg)
const D1 = -0.5         # D-term coefficient (Dg0)
const mD1 = 2.0         # D-term mass scale (MDg)
const include_n3 = false # Whether to include n=3 D-term

# Integration tolerance - reduced for speed
const INTEGRATION_TOL = 1e-4  # Further reduced for speed

# PERFORMANCE OPTIMIZATIONS:
# 1. Reduced integration tolerance (1e-4 vs 1e-8) - ~5-10x speedup
# 2. Pre-computed beta function in f_g - avoids repeated computation
# 3. Thread-based parallelism for inner calculations
# 4. Sequential parameter loop for progress tracking
# 5. CRITICAL: Reduced grid size to avoid nested integration explosion

function D_g(x::Float64, xi::Float64, t::Float64, D1::Float64, mD1::Float64, include_n3::Bool=false, D3::Float64=0.0, mD3::Float64=1.0)
    if abs(x) >= abs(xi)
        return 0.0  # outside ERBL region
    end
    
    z = x / xi
    # n=1 term (C0^{5/2} = 1)
    tripole1 = 1 / (1 - t/mD1^2)^3
    Dval = D1 * (1 - z^2)^2 * tripole1
    
    # n=3 term (optional)
    if include_n3 && D3 != 0.0
        tripole3 = 1 / (1 - t/mD3^2)^3
        C2 = gegenbauer(2, 2.5, z)  # C_{2}^{5/2}(z)
        Dval += D3 * (1 - z^2)^2 * C2 * tripole3
    end
    
    return Dval * 3/2 * 5/4
end

function pi_g(alpha::Float64, beta_abs::Float64)
    num = ((1 - beta_abs)^2 - alpha^2)^2
    denom = (1 - beta_abs)^5
    return denom > 0 ? (15/16) * num / denom : 0.0
end

function f_g(beta_abs::Float64, t::Float64, alpha::Float64, beta::Float64, norm::Float64, m::Float64)
    denom = (1 - t/m^2)^3
    
    if beta_abs == 0.0 && alpha <= 0  # regulator
        return 0.0
    end
    
    # Pre-compute beta function to avoid repeated computation
    beta_func = SpecialFunctions.beta(2+alpha, 1+beta)
    return norm/beta_func * (beta_abs^alpha) * ((1 - beta_abs)^beta) / denom
end

function H_g_DD(x::Float64, xi::Float64, t::Float64, H_alpha::Float64, H_beta::Float64, H_norm::Float64, H_m::Float64)
    if abs(xi) < 1e-12
        beta_abs = abs(x)
        alpha_min, alpha_max = -1 + beta_abs, 1 - beta_abs
        integral, _ = quadgk(alpha -> pi_g(alpha, beta_abs), alpha_min, alpha_max, rtol=INTEGRATION_TOL)
        return beta_abs * f_g(beta_abs, t, H_alpha, H_beta, H_norm, H_m) * integral
    end
    
    function integrand(beta::Float64)
        beta_abs = abs(beta)
        alpha = (x - beta) / xi
        if abs(alpha) > 1 - beta_abs
            return 0.0
        end
        return (beta_abs / abs(xi)) * pi_g(alpha, beta_abs) * f_g(beta_abs, t, H_alpha, H_beta, H_norm, H_m)
    end
    
    result, _ = quadgk(integrand, -1.0, 1.0, rtol=INTEGRATION_TOL)
    return result
end

function H_g(x::Float64, xi::Float64, t::Float64, H_alpha::Float64, H_beta::Float64, H_norm::Float64, H_m::Float64, D1::Float64, mD1::Float64, include_n3::Bool=false, D3::Float64=0.0, mD3::Float64=1.0)
    HDD = H_g_DD(x, xi, t, H_alpha, H_beta, H_norm, H_m)
    
    Dterm = abs(xi) * D_g(x, xi, t, D1, mD1, include_n3, D3, mD3)
    
    return HDD + Dterm
end

function E_g(x::Float64, xi::Float64, t::Float64, D1::Float64, mD1::Float64, include_n3::Bool=false, D3::Float64=0.0, mD3::Float64=1.0)
    Dterm = abs(xi) * D_g(x, xi, t, D1, mD1, include_n3, D3, mD3)
    return abs(x) < abs(xi) ? -Dterm : 0.0
end

function HCFF(xi::Float64, t::Float64, H_alpha::Float64, H_beta::Float64, H_norm::Float64, H_m::Float64, D1::Float64, mD1::Float64, include_n3::Bool=false, D3::Float64=0.0, mD3::Float64=1.0)
    function integrand(x::Float64)
        if abs(x - xi) < 1e-8 || abs(x + xi) < 1e-8
            return 0.0
        end
        return H_g(x, xi, t, H_alpha, H_beta, H_norm, H_m, D1, mD1, include_n3, D3, mD3) * (1/(x + xi) - 1/(x - xi)) / (2 * xi)
    end
    
    real_part, _ = quadgk(integrand, -1.0, 1.0, rtol=INTEGRATION_TOL)
    imag_part = π/xi * H_g(xi, xi, t, H_alpha, H_beta, H_norm, H_m, D1, mD1, include_n3, D3, mD3)
    return real_part + im * imag_part
end

function ECFF(xi::Float64, t::Float64, H_alpha::Float64, H_beta::Float64, H_norm::Float64, H_m::Float64, D1::Float64, mD1::Float64, include_n3::Bool=false, D3::Float64=0.0, mD3::Float64=1.0)
    function integrand(x::Float64)
        if abs(x - xi) < 1e-8 || abs(x + xi) < 1e-8
            return 0.0
        end
        return E_g(x, xi, t, D1, mD1, include_n3, D3, mD3) * (1/(x + xi) - 1/(x - xi)) / (2 * xi)
    end
    
    real_part, _ = quadgk(integrand, -1.0, 1.0, rtol=INTEGRATION_TOL)
    imag_part = π/xi * E_g(xi, xi, t, D1, mD1, include_n3, D3, mD3)
    return real_part + im * imag_part
end

function HGFF(xi::Float64, t::Float64, H_alpha::Float64, H_beta::Float64, H_norm::Float64, H_m::Float64, D1::Float64, mD1::Float64, include_n3::Bool=false, D3::Float64=0.0, mD3::Float64=1.0)
    function integrand(x::Float64)
        return H_g_DD(x, xi, t, H_alpha, H_beta, H_norm, H_m)
    end
    
    function integrandD(x::Float64)
        return abs(x) < abs(xi) ? abs(xi) * D_g(x, xi, t, D1, mD1, include_n3, D3, mD3) : 0.0
    end
    
    AGFF, _ = quadgk(integrand, 0.0, 1.0, rtol=INTEGRATION_TOL)
    DGFF, _ = quadgk(integrandD, 0.0, 1.0, rtol=INTEGRATION_TOL)
    
    return AGFF, DGFF/xi^2
end

function compute_formfactors(xi::Float64, t::Float64, H_alpha::Float64, H_beta::Float64, H_norm::Float64, H_m::Float64, D1::Float64, mD1::Float64, include_n3::Bool)
    # Compute HGFF (AGFF, DGFF)
    AGFF, DGFF = HGFF(xi, t, H_alpha, H_beta, H_norm, H_m, D1, mD1, include_n3)
    
    # Compute HCFF and ECFF
    HCFF_complex = HCFF(xi, t, H_alpha, H_beta, H_norm, H_m, D1, mD1, include_n3)
    ECFF_complex = ECFF(xi, t, H_alpha, H_beta, H_norm, H_m, D1, mD1, include_n3)
    
    return (AGFF, DGFF, real(HCFF_complex), imag(HCFF_complex), real(ECFF_complex), imag(ECFF_complex))
end

function compute_parameter_combination(H_alpha::Float64, H_beta::Float64)
    # Create output filename
    output_file = "alpha_$(H_alpha)_beta_$(H_beta).csv"
    
    # Define the lattice - 10x10 grid (reduced for speed)
    t_values = range(-10.0, 0.0, length=10)  # 10 points from -10 to 0 GeV²
    xi_values = range(0.3, 0.8, length=10)   # 10 points from 0.3 to 0.8
    
    # Create all (xi, t) pairs
    xi_t_pairs = [(xi, t) for t in t_values for xi in xi_values]
    n_pairs = length(xi_t_pairs)
    
    # Pre-allocate results array for thread safety
    results = Vector{NamedTuple}(undef, n_pairs)
    
    # Parallel computation over (xi, t) pairs using threads
    @threads for i in 1:n_pairs
        xi, t = xi_t_pairs[i]
        result = compute_formfactors(xi, t, H_alpha, H_beta, H_norm, H_m, D1, mD1, include_n3)
        results[i] = (xi=xi, t=t, AGFF=result[1], DGFF=result[2], HCFF_real=result[3], HCFF_imag=result[4], ECFF_real=result[5], ECFF_imag=result[6])
    end
    
    # Convert to DataFrame and save
    df = DataFrame(results)
    CSV.write(output_file, df)
    
    return (H_alpha, H_beta, output_file, size(df))
end

"""
Generate parameter grid for H_alpha and H_beta.
"""
function generate_parameter_grid()
    # H_alpha: 100 values from -1.9 to -0.1 (valid for Beta function: H_alpha > -2)
    H_alpha_values = range(-1.9, -0.1, length=100)
    
    # H_beta: 100 values from 3.0 to 15.0 (wide range, keeping H_beta > 3 as intended)
    H_beta_values = range(3.0, 15.0, length=100)
    
    # Create all combinations
    param_combinations = [(alpha, beta) for alpha in H_alpha_values for beta in H_beta_values]
    
    return param_combinations
end

"""
Main computation function.
"""
function run_computation(test_mode::Bool=false)
    println("GPD Form Factors Computation")
    println("=" * "="^50)
    println("Timestamp: $(now())")
    println()
    
    # Generate parameter grid
    println("Generating parameter grid...")
    param_combinations = generate_parameter_grid()
    total_combinations = length(param_combinations)
    
    if test_mode
        param_combinations = param_combinations[1:10]  # Test with 10 combinations
        total_combinations = length(param_combinations)
        println("TEST MODE: Limited to $total_combinations combinations")
    end
    
    println("Total parameter combinations: $total_combinations")
    println("H_alpha range: -1.9 to -0.1 (100 values)")
    println("H_beta range: 3.0 to 15.0 (100 values)")
    println("Each combination: 10×10 = 100 points")
    println("Total computations: $(total_combinations * 100) points")
    println()
    
    # Calculate actual ranges
    alphas = [combo[1] for combo in param_combinations]
    betas = [combo[2] for combo in param_combinations]
    alpha_min, alpha_max = minimum(alphas), maximum(alphas)
    beta_min, beta_max = minimum(betas), maximum(betas)
    alpha_count = length(unique(alphas))
    beta_count = length(unique(betas))
    
    println("Parameter ranges:")
    println("  H_alpha: $alpha_min to $alpha_max ($alpha_count values)")
    println("  H_beta: $beta_min to $beta_max ($beta_count values)")
    println()
    
    println("Using parallel computation with $(Threads.nthreads()) threads")
    println()
    
    # Run computations with sequential parameter loop and parallel inner calculations
    start_time = time()
    println("Starting computations...")
    
    # Sequential loop over parameters, parallel inner calculations
    results = []
    for (i, (H_alpha, H_beta)) in enumerate(param_combinations)
        if i % 10 == 1 || i == length(param_combinations)
            println("Processing parameter combination $i/$total_combinations (α=$H_alpha, β=$H_beta)")
        end
        result = compute_parameter_combination(H_alpha, H_beta)
        push!(results, result)
    end
    
    end_time = time()
    total_duration = end_time - start_time
    
    println()
    println("Computation completed!")
    println("Total time: $(total_duration) seconds ($(total_duration/3600) hours)")
    println("Results saved in $(length(results)) CSV files")
    
    # Save summary
    summary_df = DataFrame([
        :H_alpha => [r[1] for r in results],
        :H_beta => [r[2] for r in results],
        :output_file => [r[3] for r in results],
        :data_shape => [r[4] for r in results]
    ])
    
    summary_file = "computation_summary_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    CSV.write(summary_file, summary_df)
    println("Summary saved to: $summary_file")
    
    return results
end

"""
Main function with command-line interface.
"""
function main()
    # Parse command line arguments
    test_mode = false
    requested_threads = nothing
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--test"
            test_mode = true
        elseif arg == "--threads"
            if i + 1 <= length(ARGS)
                requested_threads = parse(Int, ARGS[i + 1])
                i += 1  # Skip the next argument since we consumed it
            else
                error("--threads requires a number")
            end
        end
        i += 1
    end
    
    # Display thread information
    current_threads = Threads.nthreads()
    if requested_threads !== nothing
        if requested_threads != current_threads
            println("Warning: Requested $requested_threads threads, but Julia was started with $current_threads threads")
            println("To use $requested_threads threads, restart Julia with: julia --threads=$requested_threads")
        end
    end
    
    println("Using $(current_threads) threads for parallel computation")
    
    if test_mode
        println("Running in TEST MODE with 10 parameter combinations")
    else
        println("Running FULL COMPUTATION with 10,000 parameter combinations")
    end
    
    println()
    
    # Run the computation
    results = run_computation(test_mode)
    
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
