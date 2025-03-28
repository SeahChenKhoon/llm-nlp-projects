# Standard library
import logging
import os
import re
from openai import OpenAI
from pathlib import Path
from typing import Dict, Any

# Third-party packages
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_python_files(directory: str) -> list[Path]:
    return list(Path(directory).rglob("*.py"))

def generate_test_prompt(file_content: str, file_path: str) -> str:
    return f"""
You're an expert Python developer. Read the following Python code and generate comprehensive pytest-style unit tests for it.

Make sure:
- All public functions and classes are tested.
- Use mock objects when needed.
- Use meaningful test function names.
- Do not include any explanations.
- Exclude any ```python code fences```.
Source file: {file_path}

Python code:
\"\"\"
{file_content}
\"\"\"
"""

def _load_env_variables() -> Dict[str, Any]:
    load_dotenv()  # Load environment variables from .env file

    return {
        "openai_api_key": os.getenv("openai_api_key"),
        "src_dir": os.getenv("src_dir"),
        "tests_dir": os.getenv("tests_dir"),
        "model": os.getenv("model"),
    }

def generate_unit_tests(model, code: str, file_path: str) -> str:
    client = OpenAI()
    prompt = generate_test_prompt(code, file_path)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def save_test_file(src_dir, tests_dir, original_path: Path, test_code: str):
    relative_path = original_path.relative_to(src_dir)
    test_path = Path(tests_dir) / relative_path
    test_path = test_path.with_name(f"test_{test_path.name}")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(test_code)
    print(f"✅ Generated test: {test_path}")

def clean_test_code(code: str) -> str:
    # Remove markdown-style code fences
    lines = code.strip().splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]

    # Remove non-code summary/comment block at the end
    # We'll look for the last function or import and trim anything after
    pattern = re.compile(r"^(def |@pytest|@patch|import |from |class )")
    last_code_index = max(
        (i for i, line in enumerate(lines) if pattern.match(line.strip())), default=len(lines) - 1
    )
    cleaned_lines = lines[: last_code_index + 1]

    return "\n".join(cleaned_lines).replace("\r", "")

def main():
    logger.info("Loading environment variables...")
    env_vars = _load_env_variables()

    python_files = get_python_files(env_vars["src_dir"])
    for file_path in python_files:
        code = file_path.read_text()
        if not code.strip():
            continue
        logging.info(f"🧠 Generating tests for {file_path}...")
        try:
            test_code = generate_unit_tests(
                model=env_vars["model"], 
                code=code, 
                file_path=str(file_path)
            )
            save_test_file(env_vars['src_dir'], env_vars['tests_dir'], file_path, test_code)
        except Exception as e:
            logging.error(f"❌ Failed to generate test for {file_path}: {e}")

if __name__ == "__main__":
    main()
