# Pre-commit Hooks Guide

## Overview

This project uses pre-commit hooks to ensure code quality and consistency. Pre-commit hooks run automatically before each commit, checking your changes against a set of rules to ensure they meet the project's standards.

## Installed Hooks

We have configured the following hooks:

1. **Code Quality**
   - Trailing whitespace removal
   - End-of-file fixing
   - Ruff (linting with auto-fixes)
   - Ruff Format (code formatting)
   - isort (import sorting)
   - Flake8 (additional linting)
   - PyUpgrade (modernize Python code)

2. **Type Checking**
   - MyPy (static type checking)

3. **Documentation**
   - Interrogate (docstring coverage)

4. **Security**
   - Check for private keys
   - Check for debug statements

5. **Tests**
   - Pytest (local testing)

## Usage

### Automatic Execution

Pre-commit hooks run automatically when you try to commit changes. If any hook fails, the commit will be aborted, allowing you to fix the issues before committing.

### Manual Execution

You can manually run the hooks using one of these methods:

1. **Using Make**

   ```bash
   # Run on staged files
   make pre-commit

   # Run on all files
   make pre-commit-all
   ```

2. **Using the Script**

   ```bash
   # Run on staged files
   ./scripts/run_pre_commit.sh

   # Run on all files
   ./scripts/run_pre_commit.sh --all
   ```

3. **Direct Command**

   ```bash
   # Run on staged files
   .venv/bin/pre-commit run

   # Run on all files
   .venv/bin/pre-commit run --all-files
   ```

## Skipping Hooks

In rare cases, you may need to bypass hooks for a specific commit:

```bash
git commit -m "Your message" --no-verify
```

**Note**: This should be used sparingly, as hooks help maintain code quality.

## Installing Pre-commit

We provide a setup script to properly install pre-commit and set up the hooks. This is the recommended way to get started:

```bash
# Run the setup script
./scripts/setup_hooks.sh
```

This script will:
1. Install pre-commit if needed
2. Uninstall any existing hooks
3. Install the hooks fresh
4. Run pre-commit once to set up all hook environments

If you prefer to do it manually:

```bash
# Install pre-commit
uv add pre-commit

# Install the hooks
.venv/bin/pre-commit install
```

## Updating Hooks

To update the pre-commit hooks to their latest versions:

```bash
.venv/bin/pre-commit autoupdate
```

## Configuration

The pre-commit configuration is defined in `.pre-commit-config.yaml` at the project root.

## Troubleshooting

If you encounter issues with the hooks:

1. **Command not found errors**: The setup script should fix this by creating isolated environments for each hook.

2. **Line length issues**: Our configuration has temporarily disabled flake8 which has strict line length requirements.

3. **Bypassing hooks temporarily**: If you need to commit despite failing hooks:
   ```bash
   git commit -m "Your message" --no-verify
   ```

4. **Reset hooks**: If hooks are not working correctly, run the setup script again:
   ```bash
   ./scripts/setup_hooks.sh
   ```
