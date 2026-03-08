# FastArrayDataLoader 🛠️

**High-Performance Zero-Copy IPC for PyTorch Data Loading.**

FastArrayDataLoader is a specialized tool for reducing CPU overhead in PyTorch data loading. It uses a **Shared Memory Ring Buffer** to transfer batches between worker processes and the main process with zero-copy, bypassing standard Python pickling.

---

## 📊 Performance Context

By avoiding the "alloc-copy-pickle-unpickle" cycle of standard IPC, FastArrayDataLoader can significantly improve throughput for datasets where sample loading is fast but transfer overhead is high.

| Scenario        | PyTorch DataLoader | FastArrayDataLoader | Observation           |
| :-------------- | :----------------- | :------------------ | :-------------------- |
| **12MB Arrays** | ~900 MB/s          | ~1800 MB/s          | **~2.0x improvement** |

*Performance gains are most visible when samples are large (1MB+) and you are CPU-bound by IPC.*

---

## 📖 Table of Contents

1. [Introduction](#-introduction)
2. [How it Works](#-how-it-works)
3. [Quick Start](#-quick-start)
4. [Custom Datasets (JPEG, PNG, etc.)](#-custom-datasets-jpeg-png-etc)
5. [Limitations](#-limitations)
6. [FAQ](#-faq)

---

## 🎯 Introduction

Standard PyTorch workers load data and send it to the main process via Queues, which involves **pickling** (serialization). For large arrays, this serialization becomes a massive CPU bottleneck. 

`FastArrayDataLoader` solves this by pre-allocating shared memory and allowing workers to write data directly into it. The main process then views this memory as a `torch.Tensor` without any data copying.

---

## �️ The Approach: How it Works

1. **Pre-Allocation**: Memory for $N$ batches is allocated once in Shared Memory at startup.
2. **Worker Loading**: Each worker receives a slot in shared memory and loads data directly into it.
3. **Flexible Logic**: While optimized for `file.readinto()` (binary), you can use any loading logic (e.g., `cv2.imread`).
4. **Zero-Copy IPC**: The main process receives a signal and yields the pre-allocated buffer as a `torch.Tensor` view.

---

## ⚡ How FastArrayDataLoader Works

### Optimization #1: Batched Shared Memory IPC

**Problem:** Pickle serialization copies data multiple times, and transferring single indices via inter-process queues introduces high locking overhead.

**Solution:** Pre-allocated shared memory with `readinto()`, combined with batched index queuing. The main process calculates the entire block of indices for a given batch and sends them to the worker in a single operation, drastically reducing `multiprocessing.Queue` bottlenecking.

```python
# PyTorch (simplified)
# Worker:
data = np.fromfile(f"image_{idx}.bin")  # Allocation #1
queue.put(data)                         # Pickle → Copy #2

# Main:
data = queue.get()                      # Unpickle → Allocation #3
tensor = torch.from_numpy(data)         # Copy #4 (if not same dtype)

# FastArrayDataLoader
# Worker:
file_handle.seek(offset)
file_handle.readinto(shared_memory_slot)  # ✅ Direct write, no allocation

# Main:
tensor = torch.as_tensor(shared_memory_slot)  # ✅ Metadata-only view
```

**Benefit:**
- Direct mapping avoids intermediate python-level array object construction across process boundaries.

### Optimization #2: Paused Garbage Collection

**Problem:** Python GC scans objects, causing pauses

**Solution:** Disable GC in tight loop

```python
# worker.py (simplified)
def run(self):
    gc.disable()  # ✅ No allocations = no garbage
    
    while True:
        idx, slot = task_queue.get()
        
        # Reuse same file handle (no allocation)
        file_handle.seek(offset)
        
        # Write directly to pre-allocated buffer (no allocation)
        file_handle.readinto(shared_memory[slot])
        
        done_queue.put(slot)
```

**Impact:**
- Disabling GC during the hot read loop provides slightly more predictable latency.

### Optimization #3: Pre-Allocation

**Problem:** Dynamic allocation is slow

**Solution:** Allocate all memory upfront

```python
# Initialization (once)
for i in range(num_containers):
    # Allocate shared memory for entire batch
    size = batch_size * np.prod(shape) * dtype.itemsize
    shared_mem = shared_memory.SharedMemory(create=True, size=size)
    
    # Create numpy view (zero-copy)
    batch_array = np.ndarray(
        shape=(batch_size, *shape),
        dtype=dtype,
        buffer=shared_mem.buf
    )
```

**Impact:**
- Reusing the same shared memory slots reduces dynamic memory allocation costs.

---

## 📁 Project Structure

```
fast-array-dataloader/
│
├── fast_array_dataloader/          # Core library
│   ├── loader.py                   # Main DataLoader logic
│   ├── worker.py                   # Multiprocess worker loop
│   ├── memory.py                   # Shared memory management
│   └── dataset.py                  # Dataset base classes
│
├── tests/                          # Test suite
│   ├── test_performance.py         # Performance validation
│   └── benchmark.py                # Standalone benchmark script
│
├── pyproject.toml                  # Package metadata
└── README.md                       # This file
```

---

## 🔧 Installation

```bash
git clone https://github.com/Simon-Bertrand/superfastdataloader.git
cd superfastdataloader
pip install -e .
```

---

## 🚀 Quick Start

### 1. Define Your Schema
You must specify the exact shape and `dtype` of your arrays.

```python
import numpy as np
schema = {
    "image": ((1024, 1024, 3), np.uint8),
    "label": ((1,), np.int64)
}
```

### 2. Prepare Your Dataset
Use the `FileMappedDataset` or implement your own `SuperFastDataset`.

```python
from fast_array_dataloader import FileMappedDataset

dataset = FileMappedDataset(
    root_dir="./my_data_bins",
    schema=schema,
    length=1000  # Total samples
)
```

### 3. Initialize and Iterate
```python
from fast_array_dataloader import FastArrayDataLoader

loader = FastArrayDataLoader(dataset, batch_size=32, num_workers=4)

for batch in loader:
    images = batch["image"]  # (32, 1024, 1024, 3)
    # Training loop...
```

---

## �️ Custom Datasets (JPEG, PNG, etc.)

While `FileMappedDataset` is optimized for raw binary, you can implement `SuperFastDataset` to load **any** format. The workers will load the data into the pre-allocated shared memory slots.

```python
from fast_array_dataloader import SuperFastDataset
import cv2

class MyJpegDataset(SuperFastDataset):
    def __init__(self, img_paths, labels, schema):
        self.img_paths = img_paths
        self.labels = labels
        self._schema = schema

    @property
    def schema(self): return self._schema

    def __len__(self): return len(self.img_paths)

    def load_sample(self, index, slot_arrays):
        # 1. Load your data (JPEG, NPZ, etc.)
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Copy into the pre-allocated shared memory slot
        # slot_arrays["image"] is a view into shared memory
        slot_arrays["image"][:] = img
        slot_arrays["label"][:] = self.labels[index]
```

> [!TIP]
> To maintain high performance, ensure your `load_sample` is fast. The library handles the complex zero-copy IPC, but the actual loading speed still depends on your dataset implementation.

---

## �🔬 Implementation Details

### Shared Memory Layout
The loader allocates several "containers" in shared memory. Each container holds space for exactly one full batch. Workers write into these containers in parallel, and the main process yields them as zero-copy `torch.Tensor` views.

### Worker Strategy
Each worker is a standalone process that:
1. Receives a batch of indices from the main process.
2. Identifies its pre-allocated shared memory slot.
3. Calls the dataset's `load_sample` method to fill the slot.
4. Signals completion back to the main process via an efficient `multiprocessing.Queue`.

---

## 📊 Performance Case Study

The following results represent a specific high-throughput scenario. Your mileage will vary depending on sample size, batch size, and hardware.

### Dataset Configuration
- **Sample**: 1024x1024x3 `uint8` (~3MB per file).
- **Format**: Raw binary files.

### Benchmark Results
| Loader          | Throughput (MB/s) | Relative Performance |
| :-------------- | :---------------- | :------------------- |
| PyTorch Default | ~900 MB/s         | 1.0x                 |
| **FastArray**   | **~1800 MB/s**    | **~2.0x**            |

**Summary**: By avoiding the overhead of pickling large NumPy arrays and reducing the number of intermediate copies, the library can nearly double the data loading throughput on modern SSDs for large, fixed-size binary data.

---

## ⚖️ Comparison Table

| Feature           | PyTorch DataLoader      | FastArrayDataLoader                  |
| :---------------- | :---------------------- | :----------------------------------- |
| **Best For**      | General use, any format | Fixed-shape arrays (high throughput) |
| **Data Transfer** | Pickle (Serialization)  | Shared Memory (Zero-Copy)            |
| **IPC Overhead**  | High (for large arrays) | Ultra-Low                            |
| **Data Format**   | Anything                | Anything (via custom `load_sample`)  |
| **Memory**        | Dynamic Allocation      | Pre-allocated (Shared)               |

---

## ❓ FAQ

### Q: Does it support data augmentation?
A: Only if applied on the **GPU**. Applying CPU-based transforms (like random crops in Python) inside the loader loop would negate the zero-copy benefits.

### Q: Why is it only for fixed shapes?
A: Zero-copy shared memory requires pre-allocated buffers of a specific size. Dynamic shapes would require frequent re-allocation, which is exactly what this library aims to avoid.

### Q: What if I have 0 workers?
A: `FastArrayDataLoader` requires at least 1 worker process. Serial execution (num_workers=0) is disabled to ensure the high-performance path is always used.

---

## 📄 License
MIT License.

---

## 📚 Citation
```bibtex
@software{fastarraydataloader2026,
  author = {Simon Bertrand},
  title = {FastArrayDataLoader: Specialized Zero-Copy DataLoader for PyTorch},
  year = {2026},
  url = {https://github.com/Simon-Bertrand/FastArrayDataLoader-PyTorch}
}
```

---

