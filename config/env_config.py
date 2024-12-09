from dotenv import load_dotenv
import os

class EnvConfig:
    load_dotenv()
    AZ_OAI_BASE = os.getenv("AZ_OAI_BASE")
    AZ_OPENAI_API_KEY = os.getenv("AZ_OPENAI_API_KEY")
    AZ_OAI_VERSION = os.getenv("AZ_OAI_VERSION")
    AZ_OAI_DEPLOYMENT = os.getenv("AZ_OAI_DEPLOYMENT")
    PDF_PATH = os.getenv("PDF_PATH")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
@classmethod
def validate(cls):
    required_vars = ["AZ_OAI_BASE", "AZ_OPENAI_API_KEY", "AZ_OAI_VERSION", "AZ_OAI_DEPLOYMENT"]
    missing = [var for var in required_vars if not getattr(cls, var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")