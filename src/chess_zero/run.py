"""
Main entry point for running from command line.
"""

import os
import sys
import multiprocessing as mp

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)


def main():
    from chess_zero import manager
    manager.start()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    sys.setrecursionlimit(10000)
    main()