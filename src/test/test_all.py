#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import inspect
from EvalTest import EvalTest

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0, parentdir)

from utils import run_test


def main():
    run_test(EvalTest,
             "\n=== Running tests the eval module ===\n")

if __name__ == "__main__":
    main()
