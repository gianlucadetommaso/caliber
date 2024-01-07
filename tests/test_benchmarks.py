import os

import pytest


@pytest.mark.parametrize("path", ["benchmarks/binary_classification.py"])
def test_example(path):
    exit_code = os.system(f"python {path}")
    assert not exit_code
