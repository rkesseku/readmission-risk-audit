import pytest

from readmit.synthetic import make_synthetic


@pytest.fixture(scope="session")
def raw():
    return make_synthetic(3000, seed=1)
