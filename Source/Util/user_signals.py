# coding: utf-8
'''
    This file handles the SIGINT exception.

    @author: Michael Bühler (buehler_michael@bluewin.ch)

'''


import signal
import psutil
import os
import sys

class GracefullExit():
    def __init__(self, parent = None):
        self.exit = False

        # Terminate all children processes
        def handle_exit_all_processes(signum, frame):
            self.exit = True
            pid = os.getpid()
            parent = psutil.Process(pid)
            children = parent.children()
            for child in children:
                os.kill(child.pid, signal.SIGINT)

            print("Terminating all Processes gracefully. Please wait a moment.\n")

        # Terminate process
        def handle_exit(signum, frame):
            self.exit = True

        if parent is not None:
            signal.signal(signal.SIGINT, handle_exit_all_processes)
        else:
            signal.signal(signal.SIGINT, handle_exit)
