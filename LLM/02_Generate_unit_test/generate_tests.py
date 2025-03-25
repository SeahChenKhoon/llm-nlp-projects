import os
import openai
from pathlib import Path

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly: "sk-..."

SRC_DIR = "src"
TESTS_DIR = "tests"
MODEL = "gpt-4"  # You can use "gpt-4-turbo" if you have access

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

Source file: {file_path}

Python code:
\"\"\"
{file_content}
\"\"\"
"""

def generate_unit_tests(code: str, file_path: str) -> str:
    prompt = generate_test_prompt(code, file_path)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def save_test_file(original_path: Path, test_code: str):
    relative_path = original_path.relative_to(SRC_DIR)
    test_path = Path(TESTS_DIR) / relative_path
    test_path = test_path.with_name(f"test_{test_path.name}")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(test_code)
    print(f"✅ Generated test: {test_path}")

def main():
    python_files = get_python_files(SRC_DIR)
    for file_path in python_files:
        code = file_path.read_text()
        if not code.strip():
            continue
        print(f"🧠 Generating tests for {file_path}...")
        try:
            test_code = generate_unit_tests(code, str(file_path))
            save_test_file(file_path, test_code)
        except Exception as e:
            print(f"❌ Failed to generate test for {file_path}: {e}")

if __name__ == "__main__":
    main()
