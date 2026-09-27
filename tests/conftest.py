import pytest

from pdtoolkit.config import Config
from pdtoolkit.data import generate


@pytest.fixture
def config():
    cfg = Config()
    cfg.data.n_per_year = 500
    cfg.models.logistic_c = [1.0]
    cfg.models.boosting_leaves = [7]
    cfg.models.boosting_iterations = 25
    cfg.validation.bootstrap_samples = 20
    return cfg


@pytest.fixture
def data(config):
    return generate(config)
