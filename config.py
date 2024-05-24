from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    MAX_TOKENS = os.getenv("MAX_TOKENS", 100)
    TEMPERATURE = os.getenv("TEMPERATURE", 1.0)
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
