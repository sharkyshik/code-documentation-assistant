"""
Sample calculator module for testing ingestion.
"""


class Calculator:
    """A simple calculator class with basic operations."""

    def __init__(self, precision: int = 2):
        """
        Initialize the calculator.

        Args:
            precision: Number of decimal places for rounding results
        """
        self.precision = precision
        self.history = []

    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        result = round(a + b, self.precision)
        self._record_operation("add", a, b, result)
        return result

    def subtract(self, a: float, b: float) -> float:
        """
        Subtract b from a.

        Args:
            a: Number to subtract from
            b: Number to subtract

        Returns:
            Difference of a and b
        """
        result = round(a - b, self.precision)
        self._record_operation("subtract", a, b, result)
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = round(a * b, self.precision)
        self._record_operation("multiply", a, b, result)
        return result

    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.

        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = round(a / b, self.precision)
        self._record_operation("divide", a, b, result)
        return result

    def _record_operation(self, op: str, a: float, b: float, result: float):
        """Record an operation in history."""
        self.history.append({
            "operation": op,
            "operands": (a, b),
            "result": result
        })

    def get_history(self) -> list:
        """Get the operation history."""
        return self.history.copy()

    def clear_history(self):
        """Clear the operation history."""
        self.history = []


def create_calculator(precision: int = 2) -> Calculator:
    """
    Factory function to create a calculator instance.

    Args:
        precision: Decimal precision for calculations

    Returns:
        A new Calculator instance
    """
    return Calculator(precision=precision)


def quick_add(a: float, b: float) -> float:
    """Convenience function for quick addition."""
    return a + b


def quick_multiply(a: float, b: float) -> float:
    """Convenience function for quick multiplication."""
    return a * b
