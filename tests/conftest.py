import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def xa():
    from caf.structure import AState, XState
    x = XState()
    return x, AState(x)
