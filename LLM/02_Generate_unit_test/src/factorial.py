def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer n.

    Parameters:
        n (int): A non-negative integer.

    Returns:
        int: The factorial of n (n!).

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result