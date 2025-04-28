"""Tests for the code sandbox implementation."""

import pytest

from app.adapters.sandbox import CodeSandbox


@pytest.fixture
def sandbox() -> CodeSandbox:
    """Create a test sandbox."""
    return CodeSandbox()


def test_validate_safe_code(sandbox: CodeSandbox) -> None:
    """Test validating safe code."""
    # Simple arithmetic
    code = """
    a = 1 + 2
    b = a * 3
    result = b - 1
    """
    is_valid, error = sandbox.validate_code(code)
    assert is_valid
    assert error == ""

    # Pandas operations
    code = """
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df.mean()
    """
    is_valid, error = sandbox.validate_code(code)
    assert is_valid
    assert error == ""

    # Numpy operations
    code = """
    import numpy as np
    arr = np.array([1, 2, 3])
    result = np.mean(arr)
    """
    is_valid, error = sandbox.validate_code(code)
    assert is_valid
    assert error == ""


def test_validate_unsafe_code(sandbox: CodeSandbox) -> None:
    """Test validating unsafe code."""
    # System command execution
    code = """
    import os
    os.system('rm -rf /')
    """
    is_valid, error = sandbox.validate_code(code)
    assert not is_valid
    assert "Access to os.system is not allowed" in error

    # File operations
    code = """
    with open('/etc/passwd', 'r') as f:
        data = f.read()
    """
    is_valid, error = sandbox.validate_code(code)
    assert not is_valid

    # Import subprocess
    code = """
    import subprocess
    subprocess.run(['ls', '-la'])
    """
    is_valid, error = sandbox.validate_code(code)
    assert not is_valid
    assert "Import of module 'subprocess' is not allowed" in error

    # Import requests
    code = """
    import requests
    response = requests.get('https://example.com')
    """
    is_valid, error = sandbox.validate_code(code)
    assert not is_valid
    assert "Import of module 'requests' is not allowed" in error


def test_execute_safe_code(sandbox: CodeSandbox) -> None:
    """Test executing safe code."""
    # Simple arithmetic
    code = """
    a = 1 + 2
    b = a * 3
    result = b - 1
    """
    result = sandbox.execute_code(code)
    assert result["success"]
    assert result["error"] is None
    assert result["locals"]["result"] == 8

    # Pandas operations
    code = """
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = df.mean()
    """
    result = sandbox.execute_code(code)
    assert result["success"]
    assert result["error"] is None
    assert "result" in result["locals"]
    assert result["locals"]["result"]["a"] == 2.0
    assert result["locals"]["result"]["b"] == 5.0


def test_execute_unsafe_code(sandbox: CodeSandbox) -> None:
    """Test executing unsafe code."""
    # System command execution
    code = """
    import os
    os.system('echo "test"')
    """
    result = sandbox.execute_code(code)
    assert not result["success"]
    assert "Access to os.system is not allowed" in result["error"]

    # Import subprocess
    code = """
    import subprocess
    result = subprocess.run(['echo', 'test'], capture_output=True)
    """
    result = sandbox.execute_code(code)
    assert not result["success"]
    assert "Import of module 'subprocess' is not allowed" in result["error"]


def test_execute_code_with_syntax_error(sandbox: CodeSandbox) -> None:
    """Test executing code with syntax errors."""
    code = """
    a = 1 +
    b = 2
    """
    result = sandbox.execute_code(code)
    assert not result["success"]
    assert "Syntax error" in result["error"]


def test_execute_code_with_runtime_error(sandbox: CodeSandbox) -> None:
    """Test executing code with runtime errors."""
    code = """
    a = 1 / 0
    """
    result = sandbox.execute_code(code)
    assert not result["success"]
    assert "division by zero" in result["error"]
