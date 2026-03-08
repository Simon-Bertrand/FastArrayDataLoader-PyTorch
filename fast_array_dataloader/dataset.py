from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
import os
import json

class SuperFastDataset(ABC):
    """
    Abstract Base Class for datasets compatible with FastArrayDataLoader.
    Unlocks high-performance Zero-Copy loading by enforcing a strict schema.
    """
    
    @property
    @abstractmethod
    def schema(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        """
        Returns the schema of the samples.
        Format: {'key_name': ((shape), numpy_dtype)}
        Example: {'image': ((1024, 1024, 3), np.uint8)}
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def load_sample(self, index: int, slot_arrays: Dict[str, np.ndarray]):
        """
        Loads a single sample into the provided shared memory slot arrays.
        """
        pass


class FileMappedDataset(SuperFastDataset):
    """
    Concrete implementation where each sample component is stored in a separate binary file.
    
    Structure expectation:
        root_dir/
            {key}_0.bin
            {key}_1.bin
            ...
    
    This strict structure allows the worker to predict file paths without overhead.
    """
    def __init__(self, root_dir: str, schema: Dict[str, Tuple[Tuple[int, ...], np.dtype]], length: int, file_pattern: str = "{root}/{key}_{index}.bin"):
        self._root_dir = root_dir
        self._schema = schema
        self._length = length
        self.file_pattern = file_pattern
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Data directory not found: {root_dir}")

    @property
    def schema(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        return self._schema

    @property
    def root_dir(self) -> str:
        return self._root_dir

    def __len__(self) -> int:
        return self._length

    def load_sample(self, index: int, slot_arrays: Dict[str, np.ndarray]):
        root_path = self.root_dir.rstrip('/\\')
        for key in self._schema.keys():
            path = self.file_pattern.format(root=root_path, key=key, index=index)
            with open(path, "rb") as f:
                f.readinto(slot_arrays[key].data)

