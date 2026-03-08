import gc
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import OrderedDict

class LoaderWorker(mp.Process):
    """
    Dedicated worker process for zero-copy data loading.
    Disables GC during the main loop for maximum throughput.
    """
    def __init__(self, configs, task_queue, done_queue, dataset):
        super().__init__()
        self.configs = configs
        self.task_queue = task_queue
        self.done_queue = done_queue
        self.dataset = dataset

    def run(self):
        # Optimization: Disable GC to prevent stop-the-world pauses in tight loops
        gc.disable()
        
        shms = []
        views = []
        
        # 1. Re-attach to shared memory blocks
        try:
            # The dataset object is already instantiated (pickled from main process)
            dataset = self.dataset
            
            for conf in self.configs:
                batch_view = {}
                for key, (name, shape, dtype) in conf.items():
                    shm = shared_memory.SharedMemory(name=name)
                    shms.append(shm)
                    # Reconstruct numpy view from shared buffer
                    batch_view[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                views.append(batch_view)

            # Localize methods to avoid attribute lookup overhead
            get_task = self.task_queue.get
            put_done = self.done_queue.put
            load_sample = dataset.load_sample

            while True:
                task = get_task()
                if task is None: 
                    break
                
                indices_batch, container_index = task
                
                try:
                    current_view = views[container_index]
                    for batch_slot, sample_index in enumerate(indices_batch):
                        # Use dataset's custom loading logic
                        # We pass a view that pointed to the specific slot in shared memory
                        slot_arrays = {
                            key: arr[batch_slot] 
                            for key, arr in current_view.items()
                        }
                        load_sample(sample_index, slot_arrays)

                    # Normal completion for entire batch chunk
                    put_done((container_index, len(indices_batch), None))
                
                except Exception as e:
                    # Error propagation
                    # If error occurs, we still return the batch slot so loader can fail gracefully
                    put_done((container_index, 0, e))
        
        finally:
            # Cleanup resources on exit
            for s in shms: 
                s.close()
            gc.enable()
