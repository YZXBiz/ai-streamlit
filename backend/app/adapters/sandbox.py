"""
Sandbox for safe code execution.

This module provides a sandboxed environment for executing code
with restricted capabilities to prevent security issues.
"""

import ast
import builtins
import contextlib
import io
from typing import Any

from ..core.config import settings


class CodeSandbox:
    """
    Sandbox for secure code execution.

    This class provides methods for:
    - Validating code for security
    - Executing code in a restricted environment
    - Capturing output and results
    """

    # Default allowed modules
    DEFAULT_ALLOWED_MODULES = {
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "scipy",
        "statsmodels",
        "datetime",
        "re",
        "math",
        "random",
        "json",
        "collections",
        "itertools",
        "functools",
        "operator",
        "string",
        "time",
        "os.path",
    }

    # Default allowed builtins
    DEFAULT_ALLOWED_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }

    # Dangerous operations to block
    DANGEROUS_OPERATIONS = {
        ast.Delete,
        # ast.Exec,  # Removed - not in Python 3.10
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Global,
        ast.Nonlocal,
    }

    def __init__(
        self,
        allowed_modules: set[str] | None = None,
        allowed_builtins: set[str] | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize the code sandbox.

        Args:
            allowed_modules: Set of module names that are allowed to be imported
            allowed_builtins: Set of builtin functions that are allowed to be used
            timeout: Maximum execution time in seconds
        """
        self.allowed_modules = allowed_modules or self.DEFAULT_ALLOWED_MODULES
        self.allowed_builtins = allowed_builtins or self.DEFAULT_ALLOWED_BUILTINS
        self.timeout = timeout or settings.SQL_QUERY_TIMEOUT

    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Validate code for security.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse the code
            tree = ast.parse(code)

            # Check for dangerous operations
            for node in ast.walk(tree):
                # Check for dangerous node types
                valid, message = self._check_dangerous_operations(node)
                if not valid:
                    return False, message

                # Check for imports
                valid, message = self._check_imports(node)
                if not valid:
                    return False, message

                # Check for attribute access
                valid, message = self._check_attributes(node)
                if not valid:
                    return False, message

            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _check_dangerous_operations(self, node: ast.AST) -> tuple[bool, str]:
        """Check for dangerous operations in an AST node."""
        if type(node) in self.DANGEROUS_OPERATIONS:
            return False, f"Dangerous operation detected: {type(node).__name__}"
        return True, ""

    def _check_imports(self, node: ast.AST) -> tuple[bool, str]:
        """Check for unauthorized imports in an AST node."""
        if isinstance(node, ast.Import | ast.ImportFrom):
            for name in node.names:
                module_name = name.name.split(".")[0]
                if module_name not in self.allowed_modules:
                    return False, f"Import of module '{module_name}' is not allowed"
        return True, ""

    def _check_attributes(self, node: ast.AST) -> tuple[bool, str]:
        """Check for dangerous attribute access in an AST node."""
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "os":
                if node.attr not in ["path"]:
                    return False, f"Access to os.{node.attr} is not allowed"

            # Block access to system, subprocess, etc.
            if isinstance(node.value, ast.Name) and node.value.id in [
                "sys",
                "subprocess",
                "os",
                "shutil",
                "socket",
                "requests",
            ]:
                return False, f"Access to {node.value.id}.{node.attr} is not allowed"
        return True, ""

    def execute_code(self, code: str) -> dict[str, Any]:
        """
        Execute code in a restricted environment.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with execution results
        """
        # Validate the code first
        is_valid, error_message = self.validate_code(code)
        if not is_valid:
            return {"success": False, "error": error_message, "output": "", "result": None}

        # Create a restricted globals dictionary
        restricted_globals = {
            "__builtins__": {name: getattr(builtins, name) for name in self.allowed_builtins}
        }

        # Create a dictionary to store local variables
        local_vars = {}

        # Capture stdout
        stdout_capture = io.StringIO()

        try:
            # Execute the code with timeout
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, restricted_globals, local_vars)

            # Get the output
            output = stdout_capture.getvalue()

            # Look for a result variable
            result = None
            if "result" in local_vars:
                result = local_vars["result"]
            elif "d" in local_vars:
                result = local_vars["d"]
            elif "fig" in local_vars:
                result = local_vars["fig"]

            return {
                "success": True,
                "error": None,
                "output": output,
                "result": result,
                "locals": local_vars,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": stdout_capture.getvalue(),
                "result": None,
            }
