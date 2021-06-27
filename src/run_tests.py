#!/usr/bin/python3
"""
Run all unit and integration tests and report the results.

Copyright 2020, 2021 Jerrad Michael Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import io
import re
import os
import sys
import enum
import time
import unittest
import subprocess
import argparse
from pathlib import Path

from coverage import Coverage

# Path to the root of the git repository.
GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
SRC_PATH = GIT_ROOT / Path('src')
TESTS_PATH = SRC_PATH / Path('tests')
UNIT_PATH = TESTS_PATH / Path('unit')
INTEGRATION_PATH = TESTS_PATH / Path('integration')
SYSTEM_PATH = TESTS_PATH / Path('system')

# The minimum coverage percentage required for the coverage test to pass.
MIN_COVERAGE_PERCENT = 100


class Verdict(enum.Enum):
    """
    Enumerates possible testcase verdicts.

    """

    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    ERROR = enum.auto()
    SKIPPED = enum.auto()
    EXPECTED_FAILURE = enum.auto()
    UNEXPECTED_SUCCESS = enum.auto()


def main(argv):
    cl_args = parse_command_line(argv)
    start_time = time.time()
    coverage_files = get_files_with_extension(SRC_PATH,
                                              '.py',
                                              exclude=['tests',
                                                       'run_tests.py',
                                                       'run_linters.py'])

    coverage = Coverage()
    coverage.start()
    unit_testsuite = unittest.defaultTestLoader.discover(UNIT_PATH,
                                                         top_level_dir=SRC_PATH)

    integration_testsuite = unittest.defaultTestLoader.discover(INTEGRATION_PATH,
                                                                top_level_dir=SRC_PATH)

    testcases = extract_tests(unit_testsuite) + extract_tests(integration_testsuite)
    if cl_args.include_system:
        system_testsuite = unittest.defaultTestLoader.discover(SYSTEM_PATH,
                                                               top_level_dir=SRC_PATH)

        testcases += extract_tests(system_testsuite)

    verdicts = list(map(run_test, testcases))
    coverage.stop()
    coverage.save()
    if coverage_files:
        coverage_stream = io.StringIO()
        coverage_percentage = coverage.report(file=coverage_stream,
                                              include=coverage_files,
                                              show_missing=True)

        coverage_report = coverage_stream.getvalue()
        coverage_stream.close()

    else:
        # coverage not run on any file changes.
        # Set coverage percentage to the highest possible value (100).
        coverage_percentage = 100
        coverage_report = ''

    total_tests = len(verdicts)
    successes = verdicts.count(Verdict.SUCCESS)
    failures = verdicts.count(Verdict.FAILURE)
    errors = verdicts.count(Verdict.ERROR)
    skipped = verdicts.count(Verdict.SKIPPED)
    expected_failures = verdicts.count(Verdict.EXPECTED_FAILURE)
    unexpected_successes = verdicts.count(Verdict.UNEXPECTED_SUCCESS)

    report = f'\nTotal tests:             {total_tests}\n'
    report += f'Successes:               {successes}\n'
    report += f'Failures:                {failures}\n'
    report += f'Errors:                  {errors}\n'
    report += f'Skipped:                 {skipped}\n'
    report += f'Expected failures:       {expected_failures}\n'
    report += f'Unexpected successes:    {unexpected_successes}\n'
    print(report)
    print(coverage_report)
    failed = (failures
              or errors
              or unexpected_successes
              or coverage_percentage < MIN_COVERAGE_PERCENT)

    if failed:
        print('\nFinal status: FAIL')

    else:
        print('\nFinal status: SUCCESS')

    print(f'Total runtime: {time.time() - start_time:.2f}')

    return failed


def run_test(test_case):
    """
    Run a single test case and print errors/failures to stderr.

    Args
      test_case: An instance of unittest.TestCase.

    Returns
      An attribute of Verdict.

    """

    print(f'{test_case.id()}.... ', end='')
    with open(os.devnull, 'w') as null_stream:
        prev_stdout = sys.stdout
        prev_stderr = sys.stderr
        try:
            sys.stdout = null_stream
            sys.stderr = null_stream
            test_result = test_case.run()

        finally:
            sys.stdout = prev_stdout
            sys.stderr = prev_stderr

    if test_result is None:
        print('skipped')
        return Verdict.SKIPPED

    assert test_result.testsRun == 1
    if test_result.failures:
        print('failure\n')
        print(test_result.failures[0][1], file=sys.stderr)
        return Verdict.FAILURE

    elif test_result.errors:
        print('error\n')
        print(test_result.errors[0][1], file=sys.stderr)
        return Verdict.ERROR

    elif test_result.expectedFailures:
        print('expected failure')
        return Verdict.EXPECTED_FAILURE

    elif test_result.unexpectedSuccesses:
        print('unexpected success')
        return Verdict.UNEXPECTED_SUCCESS

    else:
        print('success')
        return Verdict.SUCCESS


def extract_tests(testsuite):
    """
    Extract individual TestCases from a TestSuite and return them in a list.

    """

    testsuite_components = list(testsuite)
    testcases = []
    for component in testsuite_components:
        if isinstance(component, unittest.TestCase):
            testcases.append(component)

        elif isinstance(component, unittest.TestSuite):
            testcases.extend(extract_tests(component))

        else:
            assert False

    return testcases


def get_changed_files():
    """
    Get files that changed since the last commit. If there have been no
    changes since the last commit, return the files that changed between
    the last commit and master.

    """

    git_diff = subprocess.check_output(['git', 'diff']).decode('utf-8')
    if not git_diff.strip():
        git_diff = subprocess.check_output(['git', 'diff', 'origin/master']).decode('utf-8')

    changed_files = set(str(GIT_ROOT / Path(x)) for x in re.findall(r'\+\+\+ b/(.+)', git_diff))

    return changed_files


def get_files_with_extension(path, extension, exclude=None):
    """
    Get all files with the given extension that live in `path` or any of
    its children.

    Args
      path: The path to the directory to search for matching files as either a
            string or a Path object.
      extension: The file extension to match, including the leading '.'.
      exclude: (Optional) A list of directory names not to descend into.

    Returns
      A set of path strings to all matching files within the directory at `path`.

    """

    if exclude is None:
        return set(str(x) for x in Path(path).glob(f'**/*{extension}'))

    matching_files = set()
    for child in Path(path).iterdir():
        if child.is_dir() and child.name not in exclude:
            matching_files.update(get_files_with_extension(child,
                                                           extension,
                                                           exclude=exclude))

        elif child.suffix == extension and child.name not in exclude:
            matching_files.add(str(child))

    return matching_files


def parse_command_line(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-system',
                        action='store_true',
                        help='Include system tests in the testsuites to be run.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
