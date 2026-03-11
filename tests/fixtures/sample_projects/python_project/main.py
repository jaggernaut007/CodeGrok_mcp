"""Main application module."""

from typing import List, Optional


def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


class Calculator:
    """A simple calculator with basic operations."""

    def __init__(self, precision: int = 2):
        self.precision = precision
        self.history: List[str] = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = round(a + b, self.precision)
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = round(a * b, self.precision)
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide two numbers, returns None if division by zero."""
        if b == 0:
            return None
        result = round(a / b, self.precision)
        self.history.append(f"{a} / {b} = {result}")
        return result


class DataProcessor:
    """Process and transform data."""

    def __init__(self, data: List[dict]):
        self.data = data

    def filter_by_key(self, key: str, value) -> List[dict]:
        """Filter data items by key-value match."""
        return [item for item in self.data if item.get(key) == value]

    def transform(self, func) -> List[dict]:
        """Apply transformation function to all items."""
        return [func(item) for item in self.data]
