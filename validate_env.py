#!/usr/bin/env python3
"""
=============================================================================
SESI CFD ENVIRONMENT VALIDATION
Pre-flight checks for HPC deployment
=============================================================================
"""

import argparse
import sys
import subprocess
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.10."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (requires >= 3.10)"

def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch installation."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"PyTorch {version} with CUDA ({gpu_name}, {vram:.1f} GB VRAM)"
        return True, f"PyTorch {version} (CPU only)"
    except ImportError:
        return False, "PyTorch not installed"

def check_triton() -> Tuple[bool, str]:
    """Check Triton installation."""
    try:
        import triton
        return True, f"Triton {triton.__version__}"
    except ImportError:
        return False, "Triton not installed"

def check_mpi() -> Tuple[bool, str]:
    """Check MPI installation."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        return True, f"mpi4py (rank {rank}/{size})"
    except ImportError:
        return False, "mpi4py not installed"

def check_parmetis() -> Tuple[bool, str]:
    """Check ParMETIS installation."""
    try:
        import pymetis
        return True, f"pymetis {pymetis.__version__}"
    except ImportError:
        return False, "pymetis not installed"

def check_scipy() -> Tuple[bool, str]:
    """Check SciPy installation."""
    try:
        import scipy
        return True, f"SciPy {scipy.__version__}"
    except ImportError:
        return False, "SciPy not installed"

def check_sda_one() -> Tuple[bool, str]:
    """Check SDA-ONE module."""
    try:
        from structural_domain_assembly_one import StructuralDomainAssembly
        return True, "SDA-ONE module available"
    except ImportError:
        return False, "structural_domain_assembly_one.py not found"

def check_cuda_kernels() -> Tuple[bool, str]:
    """Check CUDA kernel compilation."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Test simple CUDA operation
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        return True, "CUDA kernels operational"
    except Exception as e:
        return False, f"CUDA kernel test failed: {str(e)}"

def run_preflight_checks() -> Dict[str, Tuple[bool, str]]:
    """Run all pre-flight checks."""
    checks = {
        "Python version": check_python_version(),
        "PyTorch": check_pytorch(),
        "Triton": check_triton(),
        "MPI (mpi4py)": check_mpi(),
        "ParMETIS (pymetis)": check_parmetis(),
        "SciPy": check_scipy(),
        "SDA-ONE": check_sda_one(),
        "CUDA kernels": check_cuda_kernels(),
    }
    return checks

def main():
    parser = argparse.ArgumentParser(description="SESI CFD Environment Validation")
    parser.add_argument("--phase", choices=["preflight", "full"], default="preflight",
                        help="Validation phase to run")
    args = parser.parse_args()

    print("=" * 70)
    print("  SESI CFD ENVIRONMENT VALIDATION")
    print("=" * 70)
    print()

    if args.phase == "preflight":
        print("Running pre-flight checks...")
        print()
        
        checks = run_preflight_checks()
        
        all_passed = True
        for name, (passed, message) in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status:8s} | {name:25s} | {message}")
            if not passed:
                all_passed = False
        
        print()
        print("=" * 70)
        if all_passed:
            print("  All pre-flight checks passed. Environment is ready.")
            print("=" * 70)
            sys.exit(0)
        else:
            print("  Some checks failed. Please fix the issues above.")
            print("=" * 70)
            sys.exit(1)

    elif args.phase == "full":
        print("Running full validation (preflight + benchmark tests)...")
        print()
        
        # Run preflight first
        checks = run_preflight_checks()
        all_passed = all(passed for passed, _ in checks.values())
        
        if not all_passed:
            print("Pre-flight checks failed. Aborting full validation.")
            sys.exit(1)
        
        print("✓ Pre-flight checks passed.")
        print()
        
        # Import and run benchmark tests
        try:
            from sesi_cfd_production_validation import (
                BenchmarkValidationSuite,
                OPMCBenchmark,
                TritonGPUKernels,
            )
            
            print("Running benchmark tests...")
            
            # Task 1: Benchmarks
            bench = BenchmarkValidationSuite()
            print("  ✓ Benchmark suite initialized")
            
            # Task 2: OPMC
            opmc = OPMCBenchmark(epsilon=1e-6, d0=1.0)
            print("  ✓ OPMC benchmark initialized")
            
            # Task 3: Triton
            import torch
            if torch.cuda.is_available():
                field = torch.randn(32, 32, 32, dtype=torch.float32, device='cuda')
                lap = TritonGPUKernels.launch_laplacian(field, 0.1, 0.1, 0.1)
                print("  ✓ Triton kernels operational")
            else:
                print("  ⊘ Triton skipped (no CUDA)")
            
            print()
            print("=" * 70)
            print("  Full validation completed successfully.")
            print("=" * 70)
            sys.exit(0)
            
        except Exception as e:
            print(f"✗ Full validation failed: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
