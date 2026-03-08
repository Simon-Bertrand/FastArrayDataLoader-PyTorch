
import pytest
import sys
import os
try:
    import torch
except ImportError:
    import pytest
    pytest.skip("PyTorch is not installed in this environment. Please use 'torch_cuda' env.", allow_module_level=True)

try:
    from tests.benchmark import run_pytorch_benchmark, run_superfast_benchmark, setup_data, TOTAL_SAMPLES
except ImportError:
    # Fallback if 'tests' package is not resolved locally (e.g. running from inside tests/)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.benchmark import run_pytorch_benchmark, run_superfast_benchmark, setup_data, TOTAL_SAMPLES

# S'assurer que les données existent
@pytest.fixture(scope="module", autouse=True)
def prepare_data():
    setup_data(TOTAL_SAMPLES)


def test_speedup_vs_pytorch():
    """
    Vérifie que le FastArrayDataLoader est au moins 1.5x plus rapide que PyTorch
    sur le dataset synthétique (mode parallèle).
    """
    print("\n--- Running Parallel Comparator Test ---")
    
    # Run Baselines (Parallel)
    pt_time, pt_speed = run_pytorch_benchmark()
    sf_time, sf_speed = run_superfast_benchmark()
    
    speedup = pt_time / sf_time
    
    print(f"\nSpeedup achieved (Parallel): {speedup:.2f}x")
    
    # Critère de succès : au moins 1.5x plus rapide
    assert speedup > 1.5, f"Le speedup de {speedup:.2f}x est insuffisant (cible > 1.5x)"

def test_speedup_no_workers():
    """
    Vérifie que num_workers=0 est désactivé et lève une erreur.
    """
    print("\n--- Running Serial Error Test (num_workers=0) ---")
    
    # PyTorch fonctionne en mode série
    run_pytorch_benchmark(num_workers=0)
    
    # FastArrayDataLoader doit lever une erreur
    from tests.benchmark import run_superfast_benchmark
    with pytest.raises(ValueError, match="must have at least 1 worker"):
         run_superfast_benchmark(num_workers=0)
    
    print("\nSuccessfully caught expected ValueError for num_workers=0")


