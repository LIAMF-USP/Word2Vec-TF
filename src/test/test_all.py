#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import inspect
from MetricsTest import MetricsTest
from WrapperTest import WrapperTest

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0, parentdir)

from utils import run_test


def main():
    run_test(MetricsTest,
             "\n=== Running tests for metric functions ===\n")
    run_test(WrapperTest,
    		 "\n=== Running tests for Gensim using wrapper class ===\n")
if __name__ == "__main__":
    main()
