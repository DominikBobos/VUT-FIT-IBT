#!/usr/bin/env python

import tarfile
from multiprocessing import Queue
import multiprocessing
import threading
from Queue import Empty as QueueEmpty


class TarReader:
    @staticmethod
    def read_tar_worker(tar_reader):
        for i, r in enumerate(tar_reader._tar):
            tar_reader._queue.put((tar_reader._reader(tar_reader._tar.extractfile(r), r), r))
            if tar_reader._closed or tar_reader._max_records_to_read == i+1:
                break
        tar_reader._queue.put(None)
        #tar_reader._tar.close()

    def __init__(self, tar_file_name, reader=lambda fh, r: fh.read(), max_records_to_read=None, queue_size=1000, reader_args=()):
        self._tar = tarfile.open(tar_file_name, 'r|gz')
        self._queue = multiprocessing.Queue(maxsize=queue_size)
        self._closed = False
        self._reader = lambda fh, r: reader(fh, r, *reader_args)
        self._max_records_to_read = max_records_to_read
        threading.Thread(target=self.read_tar_worker, args=(self,)).start()

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def next(self):
        i = self._queue.get()
        if i is None or self._closed:
            raise StopIteration
        return i

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._tar.close()
            try:
                self._queue.get_nowait()
                self._queue.get_nowait() # to make sure that worker puts final None to queue and terminates
            except QueueEmpty as e:
                pass

def generate_infinit_iterators(iterator_constructor, *args, **kwargs):
    """
    Create generator of iterators
    Useful for NN training with fit_on_generator and ParallelizeIt 
    # Arguments
      iterator_constructor: constructor of iterator
      *args: arguments to be passed to constructor
      **kwargs: keyword arguments to be passed to constructor
    # Returns
      generator of new instances of iterators created by calling iterator_constructor

    # Example:
    ```python
    #this would create infinite iterator of iterators on list [0,1,2,3]
    keras.utils.io_utils.GenerateInfinitIterators(iter, [0,1,2,3])

    ```
    """
    return (iterator_constructor(*args, **kwargs) for _ in itertools.count())


class ParallelizeIt:
    """
    Parallelize iterable/iterator reading via threading/multiprocessing
    Synchronization is done via Queue object.
    # Arguments
      iterator: iterable or iterator to parallelize.
      max_records_to_read: int, maximum number of items to read from iterator.
      queue_size: int, default 1000, size of a queue buffer for preloading of items.
      threaded: boolean, default True. If True, use threading module, 
        otherwise use multiprocessing
    # Returns
      Iterator over `iterator` argument
    # Example  
    ```python
    import numpy as np

    l = np.arange(10000)
    ll = keras.utils.io_utils.ParallelizeIt(l, queue_size=2)
    for a in ll:
      print("Next is:", a)

    #for usage with model.fit_on_generator create infinit generator of ParallelizeIt objects
    import numpy as np

    l = np.arange(10000)
    ll = keras.utils.io_utils.generate_infinit_iterators(keras.utils.io_utils.ParallelizeIt, l, queue_size=2)
    for lll in ll:
      for a in lll:
        print("Next is:", a)
    ```
    """
    def read_worker(self):
        for i, r in enumerate(self._iterator):      
            if self._closed.is_set() or self._max_records_to_read == i:
                break
            self._queue.put(r)
        #put sentinel to signalize end of queue
        self._queue.put(self._sentinel)

    def __init__(self, iterator, max_records_to_read=None, queue_size=1000, threaded=True):
        self._iterator = iter(iterator)
        self._queue = multiprocessing.Queue(maxsize=queue_size)
        self._max_records_to_read = max_records_to_read
        self._threaded = threaded
        self._sentinel = 'QueueSentinel'
        if self._threaded:
            self._w = threading.Thread(target=self.read_worker, args=())
            self._w.daemon = True
            self._closed = threading.Event()
        else:
            self._w = multiprocessing.Process(target=self.read_worker, args=())
            self._closed = multiprocessing.Event()
        self._initialized = False

    def __iter__(self):
        self._open()
        return self

    def __enter__(self):
        self._open()
        return self

    def next(self):
        try:
            i = self._queue.get()
            if i == self._sentinel:
                self._close()
            else:
                return i
        except KeyboardInterrupt:
            self._close()

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()

    def _open(self):
        if self._closed.is_set():
            raise StopIteration
        if self._initialized == False:
            self._w.start()
        self._initialized = True


    def _close(self):
        if not self._closed.is_set():
            self._closed.set()
            try:
            #for gratefull shutdown on interuption
            # kick the worker to free up one space in queue and after another 
            # call to worker shutdown its thread
                self._queue.get_nowait()
            except QueueEmpty as e:
                pass
        raise StopIteration


if(__name__=="__main__"):
    pass
