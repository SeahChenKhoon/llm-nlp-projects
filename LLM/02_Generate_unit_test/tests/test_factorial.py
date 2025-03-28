import pytest
from src.factorial import factorial

def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_one():
    assert factorial(1) == 1

def test_factorial_five():
    assert factorial(5) == 120

def test_factorial_with_large_number():
    assert factorial(10) == 3628800

def test_factorial_negative_raises_value_error():
    with pytest.raises(ValueError):
        factorial(-1)

def test_factorial_with_non_integer_raises_type_error():
    with pytest.raises(TypeError):
        factorial("string")