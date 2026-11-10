"""Entry point for `streamlit run streamlit_app.py` and Streamlit Community Cloud.

Living at the repo root puts the root on sys.path, so `app` and `ui` import without
`pip install -e .` (hosted platforms only run `pip install -r requirements.txt`).
"""

from ui.streamlit_app import main

main()
