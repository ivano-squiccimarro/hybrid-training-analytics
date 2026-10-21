import pytest

from hybrid_training_analytics.ingestion import read_fit
from hybrid_training_analytics.synthetic import demo_samples, fit_bytes


@pytest.fixture
def synthetic_fit(tmp_path):
    path = tmp_path / "invented.fit"
    path.write_bytes(fit_bytes(demo_samples()))
    return path


@pytest.fixture
def run(synthetic_fit):
    return read_fit(synthetic_fit, synthetic=True)
