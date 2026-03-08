import os
import time
import multiprocessing as mp
import torch
import numpy as np
from queue import Empty

from .memory import SharedBatch
from .worker import LoaderWorker
from .dataset import FileMappedDataset, SuperFastDataset

class FastArrayDataLoader:
    """
    High-performance Dataloader using Shared Memory and Zero-Copy I/O.
    
    Args:
        dataset (SuperFastDataset): The dataset definition containing schema and loading logic.
        batch_size (int): Batch size.
        num_workers (int): Number of loading processes.
        prefetch_factor (int): Number of batches to prefetch per worker (approx).
    """
    def __init__(self, 
                 dataset, 
                 batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = True,
                 prefetch_factor: int = 4,
                 **kwargs):
        
        if not isinstance(dataset, SuperFastDataset):
            raise TypeError("Dataset must be SuperFastDataset.")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Calculate total batches
        if drop_last:
            self.total_batches = len(dataset) // batch_size
        else:
            self.total_batches = (len(dataset) + batch_size - 1) // batch_size

        self.num_containers = prefetch_factor + 1
        
        self.indices = np.arange(len(dataset), dtype=np.int64)
        
        self.task_queue = mp.Queue()
        self.done_queue = mp.Queue()
        
        self.containers = [SharedBatch(batch_size, dataset.schema) for _ in range(self.num_containers)]
        # Enhanced status tracking for partial batches
        self.status = [{'busy': False, 'filled_slots': 0, 'ready': False, 'expected_slots': 0}
                       for _ in range(self.num_containers)]
        
        configs = [c.configs for c in self.containers]
        if num_workers < 1:
            raise ValueError("FastArrayDataLoader must have at least 1 worker. Serial loading (num_workers=0) is not supported.")
        self.num_workers = num_workers
        self.workers = [LoaderWorker(configs, self.task_queue, self.done_queue, dataset)
                        for _ in range(self.num_workers)]
        for w in self.workers:
            w.start()

    def __len__(self):
        """Return the number of batches per epoch."""
        return self.total_batches

    def __iter__(self):
        self.submitted_batches = 0
        self.consumed_batches = 0
        self.status = [{'busy': False, 'filled_slots': 0, 'ready': False, 'expected_slots': 0}
                       for _ in range(self.num_containers)]
        
        self._last_worker_check = time.perf_counter()
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Clear any pending items from previous run
        while not self.done_queue.empty():
            try: self.done_queue.get_nowait()
            except Empty: break
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _feed(self):
        while self.submitted_batches < self.total_batches:
            i = self.submitted_batches % self.num_containers
            if self.status[i]['busy']:
                break
                
            self.status[i]['busy'] = True
            self.status[i]['filled_slots'] = 0
            self.status[i]['ready'] = False # Reset ready status
            
            start_index = self.submitted_batches * self.batch_size
            end_index = min(start_index + self.batch_size, len(self.indices))
            batch_len = end_index - start_index
            self.status[i]['expected_slots'] = batch_len

            indices_batch = self.indices[start_index:end_index].tolist()
            self.task_queue.put((indices_batch, i))
            self.submitted_batches += 1

    def __next__(self):
        if self.consumed_batches >= self.total_batches: 
            raise StopIteration
        
        t0 = time.perf_counter()
        
        self._feed()
        expected_container = self.consumed_batches % self.num_containers
        
        # Check if already ready from a previous poll
        if self.status[expected_container]['ready']:
            return self._pop_batch(expected_container, t0)
        
        while True:
            # Check for dead workers
            current_time = time.perf_counter()
            if current_time - self._last_worker_check > 1.0:
                for w in self.workers:
                    if not w.is_alive():
                        if w.exitcode != 0:
                            raise RuntimeError(f"Worker process {w.pid} died unexpectedly.")
                self._last_worker_check = current_time
            
            try:
                # Poll for completion signals
                res = self.done_queue.get(timeout=0.01)
                container_idx, processed_count, error = res[0], res[1], (res[2] if len(res) > 2 else None)

                if error:
                    self.stop()
                    raise error

                self.status[container_idx]['filled_slots'] += processed_count
                
                if self.status[container_idx]['filled_slots'] >= self.status[container_idx]['expected_slots']:
                    self.status[container_idx]['ready'] = True
                    
                    if container_idx == expected_container:
                        return self._pop_batch(expected_container, t0)
                    
            except Empty:
                self._feed()

    def _pop_batch(self, container_idx, t0):
        data = self.containers[container_idx].arrays
        valid_len = self.status[container_idx]['expected_slots']
        
        out_batch = {}
        for k, v in data.items():
            tensor = torch.as_tensor(v)
            if valid_len < self.batch_size:
                tensor = tensor[:valid_len]
            out_batch[k] = tensor
        
        self.status[container_idx]['busy'] = False
        self.status[container_idx]['ready'] = False
        self.consumed_batches += 1
        return out_batch, (time.perf_counter() - t0)

    def stop(self):
        """Clean shutdown of workers and memory."""
        # Send kill signal to workers
        for _ in self.workers: 
            self.task_queue.put(None)
        
        for w in self.workers: 
            if w.is_alive():
                w.join(timeout=1.0)
                if w.is_alive():
                    w.terminate()
            
        for c in self.containers: 
            c.cleanup()
            
    def __del__(self):
        try: self.stop()
        except: pass
