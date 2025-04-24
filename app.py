"""
Main application entry point.

This file serves as the entry point for the modular chatbot application.
All actual logic is in the flat_chatbot package.
"""

# Import the app from flat_chatbot directly
from flat_chatbot.app import main

if __name__ == "__main__":
    main()
