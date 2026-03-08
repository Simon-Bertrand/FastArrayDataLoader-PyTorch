import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys

# Ensure root (parent) is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_array_dataloader import FastArrayDataLoader, FileMappedDataset

# Config
DEFAULT_DATA_DIR = os.path.join("data", "normal")
DATA_DIR = DEFAULT_DATA_DIR
BATCH_SIZE = 8
# Set minimum workers because FastArrayDataLoader relies on worker processes to feed the queue.
NUM_WORKERS = 8
# Drastically reduce samples to prevent timeouts (4 * BATCH_SIZE is sufficient)
TOTAL_SAMPLES = 256
# Reduce size of benchmark image drastically to avoid massive CI generation and timeouts.
# We maintain realistic proportions but reduce resolution to evaluate overhead correctly.
SHAPE_IM1 = (1024,  1024, 3)
SHAPE_IM2 = (256, 256)

# --- Helper: Setup Data (Previously in dataloader.py) ---
def setup_data(n, data_dir=DATA_DIR):
    expected_size_im1 = np.prod(SHAPE_IM1) * 4 # float32
    expected_size_im2 = np.prod(SHAPE_IM2) * 4 # float32
    
    should_gen = False
    if not os.path.exists(data_dir):
        should_gen = True
    else:
        # Check if first sample matches expected size
        test_file = os.path.join(data_dir, "im1_0.bin")
        if not os.path.exists(test_file) or os.path.getsize(test_file) != expected_size_im1:
            print(f"Data shape change detected (expected {expected_size_im1} bytes, got {os.path.getsize(test_file) if os.path.exists(test_file) else 0}). Regenerating...")
            import shutil
            shutil.rmtree(data_dir)
            should_gen = True

    if should_gen:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        print(f"Generating {n} samples (Files) in {data_dir}...")
        for i in range(n):
            if i % 100 == 0: print(f"\rGen Files {i}/{n}", end="")
            np.full(SHAPE_IM1, float(i), dtype=np.float32).tofile(os.path.join(data_dir, f"im1_{i}.bin"))
            np.full(SHAPE_IM2, float(i), dtype=np.float32).tofile(os.path.join(data_dir, f"im2_{i}.bin"))
        print("\nFiles Generation completed.")

# --- PyTorch Baseline (Files) ---
class NaiveDataset(Dataset):
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.dtypes = {"im1": np.float32, "im2": np.float32}
        self.shapes = {"im1": SHAPE_IM1, "im2": SHAPE_IM2}

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {}
        # IM1
        with open(os.path.join(DATA_DIR, f"im1_{idx}.bin"), "rb") as f:
            data = np.frombuffer(f.read(), dtype=self.dtypes["im1"]).reshape(self.shapes["im1"])
            sample["im1"] = torch.from_numpy(data.copy())
        # IM2
        with open(os.path.join(DATA_DIR, f"im2_{idx}.bin"), "rb") as f:
            data = np.frombuffer(f.read(), dtype=self.dtypes["im2"]).reshape(self.shapes["im2"])
            sample["im2"] = torch.from_numpy(data.copy())
        return sample

def run_pytorch_benchmark(num_workers=NUM_WORKERS):
    mode_name = f"PyTorch (Workers={num_workers})"
    print(f"\n--- [{mode_name}] Démarrage du Benchmark ---")
    
    dataset = NaiveDataset(TOTAL_SAMPLES)
        
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        shuffle=False
    )
    
    latencies = []
    start_time = time.perf_counter()
    
    t_wait_start = time.perf_counter()
    for i, batch in enumerate(dataloader):
        t_wait_end = time.perf_counter()
        latencies.append(t_wait_end - t_wait_start)
        
        if i % 20 == 0:
             print(f"[{mode_name}] Batch {i:03} | Latence: {latencies[-1]*1000:.2f} ms")
        
        t_wait_start = time.perf_counter()

    total_time = time.perf_counter() - start_time
    
    # Stats (ignore warmup)
    avg_lat = np.mean(latencies[5:]) * 1000 if len(latencies) > 5 else 0
    # Use float for product to avoid overflow
    bytes_per_sample = float(np.prod(SHAPE_IM1) * 4 + np.prod(SHAPE_IM2) * 4)
    total_gb = (bytes_per_sample * TOTAL_SAMPLES) / (1024.0**3)
    
    print(f"\n--- [{mode_name}] RÉSULTATS ---")
    print(f"Temps total     : {total_time:.2f} s")
    print(f"Débit moyen     : {(total_gb * 1024.0) / total_time:.2f} Mo/s")
    print(f"Latence Moyenne : {avg_lat:.3f} ms")
    
    return total_time, (total_gb * 1024.0) / total_time

def run_superfast_benchmark(num_workers=NUM_WORKERS):
    mode_name = f"Superfast (Workers={num_workers})"
    print(f"\n--- [{mode_name}] Démarrage du Benchmark ---")
    schema = {"im1": (SHAPE_IM1, np.float32), "im2": (SHAPE_IM2, np.float32)}

    sf_dataset = FileMappedDataset(DATA_DIR, schema, TOTAL_SAMPLES)
    
    print("MEASURE...")
    loader = FastArrayDataLoader(dataset=sf_dataset, batch_size=BATCH_SIZE, num_workers=num_workers)
    
    latencies = []
    start_time = time.perf_counter()
    
    for i, (batch, lat) in enumerate(loader):
        latencies.append(lat)
        if i % 20 == 0:
            print(f"[{mode_name}] Batch {i:03} | Latence: {lat*1000:.2f} ms")

    total_time = time.perf_counter() - start_time
    loader.stop()

    avg_lat = np.mean(latencies[5:]) * 1000 if len(latencies) > 5 else 0
    bytes_per_sample = float(np.prod(SHAPE_IM1) * 4 + np.prod(SHAPE_IM2) * 4)
    total_gb = (bytes_per_sample * TOTAL_SAMPLES) / (1024.0**3)
    
    print(f"\n--- [{mode_name}] RÉSULTATS ---")
    print(f"Temps total     : {total_time:.2f} s")
    print(f"Débit moyen     : {(total_gb * 1024.0) / total_time:.2f} Mo/s")
    return total_time, (total_gb * 1024.0) / total_time

if __name__ == "__main__":
    setup_data(TOTAL_SAMPLES)
    
    print("RUNNING BENCHMARKS")
    
    # Multiprocessing
    print("\n>>> MULTIPROCESSING MODE <<<")
    pt_time_mp, pt_speed_mp = run_pytorch_benchmark(NUM_WORKERS)
    sf_time_mp, sf_speed_mp = run_superfast_benchmark(NUM_WORKERS)
    
    print(f"PARALLEL | PyTorch   : {pt_speed_mp:.2f} Mo/s ({pt_time_mp:.2f}s)")
    print(f"PARALLEL | Superfast : {sf_speed_mp:.2f} Mo/s ({sf_time_mp:.2f}s) -> {pt_time_mp/sf_time_mp:.2f}x Speedup")
    print(f"{'='*60}")
