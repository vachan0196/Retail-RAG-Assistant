import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("USE_OPENAI:", os.getenv("USE_OPENAI"))
print("DOCS_DIR:", os.getenv("DOCS_DIR"))
