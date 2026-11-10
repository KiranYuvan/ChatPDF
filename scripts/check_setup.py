"""Environment smoke test: Python version, installed packages, config, which LLM will be used.

Run from the project root:
    python -m scripts.check_setup
"""

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version

# (import name, pip distribution name)
PACKAGES = [
    ("pydantic_settings", "pydantic-settings"),
    ("pypdf", "pypdf"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("chromadb", "chromadb"),
    ("ollama", "ollama"),
    ("groq", "groq"),
    ("streamlit", "streamlit"),
]


def check_python() -> bool:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) <= (3, 12)
    status = "OK " if ok else "WARN"
    print(f"[{status}] Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    if not ok:
        print("       Recommended: 3.11 or 3.12 for reliable torch/chromadb wheels.")
    if sys.prefix == sys.base_prefix:
        print("[WARN] Not running inside a virtual environment (.venv).")
    return ok


def check_packages() -> bool:
    all_ok = True
    for module, dist in PACKAGES:
        try:
            importlib.import_module(module)
            print(f"[OK ] {dist:<26} {version(dist)}")
        except (ImportError, PackageNotFoundError) as exc:
            all_ok = False
            print(f"[FAIL] {dist:<26} {exc}")
    return all_ok


def check_config() -> bool:
    from app.config import get_settings

    try:
        s = get_settings()
    except Exception as exc:  # pydantic ValidationError, bad .env, etc.
        print(f"[FAIL] Config invalid:\n{exc}")
        return False

    s.ensure_dirs()
    print("[OK ] Config loaded:")
    for key, value in s.model_dump().items():
        print(f"       {key:<16} = {value}")  # SecretStr prints as '**********'
    return True


def check_llm() -> None:
    from app.config import get_settings
    from app.llm import resolve_provider

    s = get_settings()
    print(f"[INFO] LLM provider setting = {s.llm_provider!r} -> will use {resolve_provider(s)!r}")


if __name__ == "__main__":
    print("=== ChatPDF setup check ===")
    results = [check_python(), check_packages(), check_config()]
    check_llm()
    print("=== ALL GOOD ===" if all(results) else "=== Fix the items above ===")
    sys.exit(0 if all(results) else 1)
