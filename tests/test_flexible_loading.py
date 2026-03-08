import torch
import numpy as np
import pytest
from fast_array_dataloader import SuperFastDataset, FastArrayDataLoader

class MockFlexibleDataset(SuperFastDataset):
    """
    Simulates a dataset that doesn't load from binary files.
    Instead, it generates data on the fly or transforms it,
    simulating what a JPEG/PNG loader would do.
    """
    def __init__(self, length=100):
        self._length = length
        self._schema = {
            "image": ((32, 32, 3), np.uint8),
            "label": ((1,), np.int64)
        }

    @property
    def schema(self):
        return self._schema

    def __len__(self):
        return self._length

    def load_sample(self, index, slot_arrays):
        # Simulating loading/transforming data
        # In a real case, this could be cv2.imread(...)
        fake_img = np.full((32, 32, 3), index % 256, dtype=np.uint8)
        fake_label = np.array([index], dtype=np.int64)
        
        # Copy into shared memory slot
        slot_arrays["image"][:] = fake_img
        slot_arrays["label"][:] = fake_label

def test_flexible_loading_logic():
    dataset = MockFlexibleDataset(length=64)
    batch_size = 8
    num_workers = 2
    
    loader = FastArrayDataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    
    batches_count = 0
    for idx, (batch, latency) in enumerate(loader):
        batches_count += 1
        
        # Verify data correctness
        # For batch 0, samples are 0, 1, ..., 7
        expected_labels = np.arange(idx * batch_size, (idx + 1) * batch_size)
        actual_labels = batch["label"].squeeze(-1).numpy()
        
        assert np.array_equal(actual_labels, expected_labels)
        assert batch["image"].shape == (batch_size, 32, 32, 3)
        assert batch["image"].dtype == torch.uint8
        
        # Check image content
        for i in range(batch_size):
            val = (idx * batch_size + i) % 256
            assert torch.all(batch["image"][i] == val)

    assert batches_count == 64 // 8

if __name__ == "__main__":
    test_flexible_loading_logic()
    print("Verification test passed!")
