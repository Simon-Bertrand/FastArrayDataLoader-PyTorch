import os
import shutil
import pytest
import numpy as np
try:
    import torch
except ImportError:
    pytest.skip("PyTorch is not installed in this environment. Please use 'torch_cuda' env.", allow_module_level=True)

from fast_array_dataloader import FastArrayDataLoader, FileMappedDataset



def test_correctness_file_mapped(temp_data_dir):
    """Test data correctness and drop_last=False behavior."""
    N = 10
    schema = {"data": ((2,), np.int64)}

    # Generate data: data_i = [i, i*2]
    for i in range(N):
        arr = np.array([i, i*2], dtype=np.int64)
        with open(os.path.join(temp_data_dir, f"data_{i}.bin"), "wb") as f:
            f.write(arr.tobytes())

    dataset = FileMappedDataset(temp_data_dir, schema, N)

    # Test with drop_last=False, batch_size=3
    # Expected batches: [0,1,2], [3,4,5], [6,7,8], [9]
    # Note: We must verify inside the context or clone, because exiting the context
    # unlinks the shared memory, causing segfaults if accessed later.
    with FastArrayDataLoader(dataset, batch_size=3, drop_last=False, num_workers=2) as loader:
        batches = []
        for i, (batch, _) in enumerate(loader):
            # Verify shapes immediately
            if i < 3:
                assert batch["data"].shape == (3, 2)
            else:
                assert batch["data"].shape == (1, 2)
                expected_last = torch.tensor([[9, 18]], dtype=torch.int64)
                assert torch.equal(batch["data"], expected_last)

            # Verify content for first batch
            if i == 0:
                expected_b0_r1 = torch.tensor([1, 2], dtype=torch.int64)
                assert torch.equal(batch["data"][1], expected_b0_r1)

            batches.append(batch)

    assert len(batches) == 4

def test_error_propagation(temp_data_dir):
    """Test that worker errors (missing file) are propagated to main process."""
    N = 5
    schema = {"data": ((1,), np.int64)}
    for i in range(N):
        arr = np.array([i], dtype=np.int64)
        with open(os.path.join(temp_data_dir, f"data_{i}.bin"), "wb") as f:
            f.write(arr.tobytes())

    dataset = FileMappedDataset(temp_data_dir, schema, N)

    # Delete a file to trigger FileNotFoundError in worker
    os.remove(os.path.join(temp_data_dir, "data_2.bin"))

    with FastArrayDataLoader(dataset, batch_size=1, num_workers=2) as loader:
        with pytest.raises(Exception) as excinfo:
            for i, (batch, _) in enumerate(loader):
                pass
        # Ensure it's not a generic crash but the actual error
        assert "No such file" in str(excinfo.value) or isinstance(excinfo.value, FileNotFoundError)

def test_custom_file_pattern(temp_data_dir):
    """Test custom file pattern support."""
    N = 3
    schema = {"data": ((1,), np.int64)}

    # naming: sample-0-data.bin
    for i in range(N):
        arr = np.array([i], dtype=np.int64)
        with open(os.path.join(temp_data_dir, f"sample-{i}-data.bin"), "wb") as f:
            f.write(arr.tobytes())

    # Pattern: {root}/sample-{index}-{key}.bin
    pattern = "{root}/sample-{index}-{key}.bin"
    dataset = FileMappedDataset(temp_data_dir, schema, N, file_pattern=pattern)

    with FastArrayDataLoader(dataset, batch_size=1, num_workers=1) as loader:
        count = 0
        for batch, _ in loader:
            count += 1
            assert batch["data"].item() == count - 1
        assert count == N

def test_ordering(temp_data_dir):
    """Test that a large dataset returns batches in the exact sequential order when shuffle=False."""
    N = 2048
    schema = {"data": ((1,), np.int64)}

    for i in range(N):
        arr = np.array([i], dtype=np.int64)
        with open(os.path.join(temp_data_dir, f"data_{i}.bin"), "wb") as f:
            f.write(arr.tobytes())

    dataset = FileMappedDataset(temp_data_dir, schema, N)

    # Use a small batch size to test the ring buffer wrap-around
    # Use 1 worker to avoid multiprocessing synchronization hangs in certain constrained CI environments,
    # since we are just testing the sequential feed logic, not IPC latency.
    batch_size = 4

    with FastArrayDataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False) as loader:
        current_expected_val = 0
        for batch, _ in loader:
            data = batch["data"]
            for j in range(data.shape[0]):
                assert data[j].item() == current_expected_val, f"Expected {current_expected_val}, got {data[j].item()}"
                current_expected_val += 1

        assert current_expected_val == N

def test_type_error_on_invalid_dataset():
    """Test that loader raises TypeError on invalid dataset types."""
    with pytest.raises(TypeError, match="Dataset must be SuperFastDataset"):
        FastArrayDataLoader("this_is_not_a_dataset", batch_size=1)

def test_shuffle_behavior(temp_data_dir):
    """Test that shuffle=True changes the order of elements."""
    N = 50
    schema = {"data": ((1,), np.int64)}

    for i in range(N):
        arr = np.array([i], dtype=np.int64)
        with open(os.path.join(temp_data_dir, f"data_{i}.bin"), "wb") as f:
            f.write(arr.tobytes())

    dataset = FileMappedDataset(temp_data_dir, schema, N)

    # Run with shuffle=False
    seq_elements = []
    with FastArrayDataLoader(dataset, batch_size=5, num_workers=1, shuffle=False) as loader:
        for batch, _ in loader:
            seq_elements.extend(batch["data"].flatten().tolist())

    # Run with shuffle=True
    shuffled_elements = []
    with FastArrayDataLoader(dataset, batch_size=5, num_workers=1, shuffle=True) as loader:
        for batch, _ in loader:
            shuffled_elements.extend(batch["data"].flatten().tolist())

    assert len(seq_elements) == N
    assert len(shuffled_elements) == N
    assert sorted(seq_elements) == sorted(shuffled_elements)
    assert seq_elements != shuffled_elements

def test_drop_last_true(temp_data_dir):
    """Test drop_last=True behavior."""
    N = 10
    schema = {"data": ((1,), np.int64)}

    for i in range(N):
        arr = np.array([i], dtype=np.int64)
        with open(os.path.join(temp_data_dir, f"data_{i}.bin"), "wb") as f:
            f.write(arr.tobytes())

    dataset = FileMappedDataset(temp_data_dir, schema, N)

    # 10 elements, batch_size=3.
    # drop_last=True means we expect 3 batches of 3 = 9 elements.
    with FastArrayDataLoader(dataset, batch_size=3, drop_last=True, num_workers=1) as loader:
        count = 0
        for batch, _ in loader:
            assert batch["data"].shape[0] == 3
            count += 3

    assert count == 9
