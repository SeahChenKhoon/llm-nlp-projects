uv init
uv add openai
uv add dotenv
uv run generate_tests.py

uv run generate_tests.py
PYTHONPATH=. uv run pytest -v tests/test_ConvertFtToMeters.py