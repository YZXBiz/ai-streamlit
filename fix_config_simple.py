#!/usr/bin/env python3
"""Script to fix the config.py file by replacing the problematic ACCESS_TOKEN_EXPIRE_MINUTES setting."""

# Read the file
with open("app/core/config.py", "r") as f:
    content = f.read()

# Replace the problematic section
new_content = content.replace(
    """    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 
        1440,
        description="Token expiry time in minutes",
    )""",
    """    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # Token expiry time in minutes""",
)

# Write back to the file
with open("app/core/config.py", "w") as f:
    f.write(new_content)

print("Config file has been fixed.")
