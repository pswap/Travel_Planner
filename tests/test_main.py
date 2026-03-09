"""
Unit tests for the main module.
"""

import unittest
from src import main


class TestMain(unittest.TestCase):
    """Test cases for main module."""

    def test_main_function_exists(self):
        """Test that main function exists."""
        self.assertTrue(callable(main.main))


if __name__ == "__main__":
    unittest.main()
