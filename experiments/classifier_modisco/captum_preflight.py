"""Strict CPU-only dependency/import and no-skipped-tests gate for Captum."""
import argparse
import importlib
from importlib import metadata
import os
from pathlib import Path
import unittest

from .common import digest, event, write_json


EXPECTED_TESTS = 82


def validate_result(result):
    if not result.wasSuccessful() or result.skipped or result.testsRun != EXPECTED_TESTS:
        raise RuntimeError(f'Expected {EXPECTED_TESTS} passing tests, no skips; '
                           f'ran={result.testsRun}, skipped={len(result.skipped)}')


def check_dependencies(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Preflight must explicitly hide CUDA')
    # A bad vendor import or missing transitive dependency must be fatal before
    # unittest has a chance to classify it as an optional dependency skip.
    from classifier_motifs.captum_deeplift import require_pinned_captum
    require_pinned_captum()
    import numpy
    import packaging
    import torch
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    overlay = {canonicalize_name(d.metadata['Name']):d
               for d in metadata.distributions(path=[str(root/'dependencies')])}
    pinned = {}
    for line in (root/'requirements.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            key = canonicalize_name(name)
            if key not in overlay or overlay[key].version != version:
                raise RuntimeError('Missing or mismatched isolated dependency: '+line)
            pinned[key] = version
    versions = dict(pinned, torch=torch.__version__, numpy=numpy.__version__, packaging=packaging.__version__)
    for distribution in overlay.values():
        for text in distribution.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({'extra':''}):
                continue
            name = canonicalize_name(req.name)
            if name not in versions or not req.specifier.contains(versions[name], prereleases=True):
                raise RuntimeError('Unsatisfied isolated dependency requirement: '+text)
    for name in ('tqdm.auto', 'matplotlib.pyplot', 'contourpy', 'cycler', 'fontTools',
                 'kiwisolver', 'pyparsing', 'dateutil', 'six', 'PIL.Image'):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to((root/'dependencies').resolve()):
            raise RuntimeError('Dependency escaped isolated directory: '+name)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    versions = check_dependencies(root)
    event('captum_dependencies_passed', versions=versions)
    suite = unittest.defaultTestLoader.discover(str(root/'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status='passed' if result.wasSuccessful() and not result.skipped
                  and result.testsRun == EXPECTED_TESTS else 'failed',
                  tests=result.testsRun, skips=result.skipped, versions=versions,
                  manifest_sha256=digest(root/'MANIFEST.sha256'))
    write_json(args.report, report)
    validate_result(result)
    event('captum_strict_preflight_passed', tests=result.testsRun, skipped=0)


if __name__ == '__main__':
    main()
