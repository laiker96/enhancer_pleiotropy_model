"""Strict no-skip CPU preflight for the isolated IG64 production package."""
import argparse
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from .common import digest, write_json, event


EXPECTED_TESTS = 71


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU preflight requires CUDA explicitly hidden')
    suite = unittest.defaultTestLoader.discover(str(args.root/'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and not result.skipped and result.testsRun == EXPECTED_TESTS
    write_json(args.report,dict(status='passed' if passed else 'failed',tests=result.testsRun,
        skips=result.skipped,manifest_sha256=digest(args.root/'MANIFEST.sha256'),
        torch_version=torch.__version__,numpy_version=np.__version__))
    if not passed:
        raise RuntimeError(f'Expected {EXPECTED_TESTS} passing tests and no skips')
    event('ig64_preflight_passed',tests=result.testsRun,skipped=0)


if __name__ == '__main__': main()
