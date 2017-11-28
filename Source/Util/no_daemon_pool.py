"""A class inheriting from pool.Pool which allows children to spawn their own threads and processes.
The implementation is based on an answer of Chris Arndt in a thread in stackoverflow:
https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic (accessed 19.10.2017)
"""

import multiprocessing
from multiprocessing.pool import Pool

class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonPool(Pool):
    Process = NoDaemonProcess
