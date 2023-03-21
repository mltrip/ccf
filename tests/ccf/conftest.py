import os

from dotenv import load_dotenv
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--influxdb", action="store_true", default=False, help="run influxdb tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "influxdb: mark test as influxdb")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--influxdb"):
        # --influxdb given in cli: do not skip influxdb tests
        return
    skip_influxdb = pytest.mark.skip(reason="need --influxdb option to run")
    for item in items:
        if "influxdb" in item.keywords:
            item.add_marker(skip_influxdb)


@pytest.fixture(scope='session', autouse=True)
def load_env():
    load_dotenv()
    