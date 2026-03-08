
def test_worker_crash_detection(temp_data_dir):
    """Test that the main process detects a killed worker."""
    import signal
    import time
    import numpy as np
    import os
    import pytest
    from fast_array_dataloader import FastArrayDataLoader, FileMappedDataset
    
    N = 500
    schema = {"data": ((1,), np.int64)}
    
    # Generate data
    for i in range(N):
        arr = np.array([i], dtype=np.int64)
        with open(os.path.join(temp_data_dir, f"data_{i}.bin"), "wb") as f:
            f.write(arr.tobytes())
            
    dataset = FileMappedDataset(temp_data_dir, schema, N)
    
    # We need a large enough number of batches so the loop runs for a bit
    with FastArrayDataLoader(dataset, batch_size=1, num_workers=2) as loader:
        # Kill one worker manually
        # Wait for workers to start
        print("Waiting for workers...")
        time.sleep(1)
        victim = loader.workers[0]
        print(f"Killing worker {victim.pid}")
        os.kill(victim.pid, signal.SIGTERM)
        
        # Iterate, expecting a crash
        print("Iterating...")
        with pytest.raises(RuntimeError) as excinfo:
            for i, _ in enumerate(loader):
                if i % 10 == 0:
                    print(f"Iteration {i}")
                time.sleep(0.1)
        
        print("Caught expected error:", excinfo.value)
        assert "died unexpectedly" in str(excinfo.value)
