import numpy as np
from multiprocessing import shared_memory
from typing import Dict, Tuple

class SharedBatch:
    """
    Manages shared memory allocation for a single batch.
    """
    def __init__(self, batch_size: int, schema: Dict[str, Tuple[Tuple, np.dtype]]):
        self.shared_memories = []
        self.arrays = {}
        self.configs = {}

        for key, (shape, dtype_spec) in schema.items():
            dtype = np.dtype(dtype_spec)
            # Full shape includes batch dimension
            full_shape = (batch_size, *shape)
            size = int(np.prod(full_shape) * dtype.itemsize)
            
            # Allocate shared memory
            shared_mem = shared_memory.SharedMemory(create=True, size=size)
            self.shared_memories.append(shared_mem)
            
            # Create numpy view
            self.arrays[key] = np.ndarray(full_shape, dtype=dtype, buffer=shared_mem.buf)
            
            # Store config for workers to reconstruct views
            self.configs[key] = (shared_mem.name, full_shape, dtype)

    def cleanup(self):
        """Close and unlink all shared memory blocks."""
        for s in self.shared_memories:
            s.close()
            try: 
                s.unlink()
            except FileNotFoundError: 
                pass
