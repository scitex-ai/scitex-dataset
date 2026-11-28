"""`scitex-dataset api list`: the shared scitex-sdk listing bound to this package."""

import json

import pytest
from click.testing import CliRunner

from scitex_dataset._cli import main


@pytest.fixture(scope="module")
def listing():
    result = CliRunner().invoke(main, ["api", "list", "--format", "json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_api_list_shows_every_declared_rest_operation(listing):
    # Arrange
    from scitex_dataset._rest import OPERATIONS

    expected = {op.operation_id for op in OPERATIONS}
    # Act
    listed = {e["operation_id"] for e in listing["entries"] if e["kind"] == "operation"}
    # Assert
    assert listed == expected


def test_every_declared_operation_has_its_mcp_twin_served(listing):
    # Arrange
    ops = [e for e in listing["entries"] if e["kind"] == "operation"]
    # Act
    served = [e["mcp_twin_served"] for e in ops]
    # Assert
    assert served and all(served)


def test_api_list_includes_the_mcp_tools_of_the_package(listing):
    # Arrange
    tools = [e for e in listing["entries"] if e["kind"] == "mcp"]
    # Act
    derived = {e["name"] for e in tools if e["derived_from_operation"]}
    # Assert
    assert "dataset_datasets_list" in derived
