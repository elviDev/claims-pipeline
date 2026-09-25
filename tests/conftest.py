import pytest

from pipeline.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    """One Spark session shared by all tests (starting Spark takes a few seconds)."""
    session = get_spark("claims-pipeline-tests")
    yield session
    session.stop()
