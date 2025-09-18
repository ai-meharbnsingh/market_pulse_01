# verify_keys.py

import os
import asyncio
from dotenv import load_dotenv

# --- Safely import provider SDKs ---
try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# --- Configuration ---
load_dotenv()

# Define the models TO TEST, aligned with our project's configuration
# We will test the default models from our adapters and other common ones.
MODELS_TO_TEST = {
    "openai": ["gpt-4o", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"],  # REMOVED claude-3-opus
    "google_gemini": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"],
}


# --- ANSI Color Codes for Output ---
class colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'


# --- Verification Functions ---

async def verify_openai(api_key: str, models: list):
    print(f"\n{colors.BLUE}--- Verifying OpenAI Key ---{colors.ENDC}")
    if not OPENAI_AVAILABLE:
        print(f"{colors.YELLOW}OpenAI SDK not installed. Skipping.{colors.ENDC}")
        return
    try:
        client = AsyncOpenAI(api_key=api_key)
        for model in models:
            print(f"Testing model: {model} ... ", end="")
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "This is a test."}],
                max_tokens=5
            )
            print(f"{colors.GREEN}SUCCESS{colors.ENDC}")
    except Exception as e:
        print(f"{colors.RED}FAILED{colors.ENDC}")
        print(f"  Error: {e.__class__.__name__}: {e}")


async def verify_anthropic(api_key: str, models: list):
    print(f"\n{colors.BLUE}--- Verifying Anthropic (Claude) Key ---{colors.ENDC}")
    if not ANTHROPIC_AVAILABLE:
        print(f"{colors.YELLOW}Anthropic SDK not installed. Skipping.{colors.ENDC}")
        return
    try:
        client = AsyncAnthropic(api_key=api_key)
        for model in models:
            print(f"Testing model: {model} ... ", end="")
            await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": "This is a test."}],
                max_tokens=5
            )
            print(f"{colors.GREEN}SUCCESS{colors.ENDC}")
    except Exception as e:
        print(f"{colors.RED}FAILED{colors.ENDC}")
        print(f"  Error: {e.__class__.__name__}: {e}")


async def verify_gemini(api_key: str, models: list):
    print(f"\n{colors.BLUE}--- Verifying Google (Gemini) Key ---{colors.ENDC}")
    if not GOOGLE_AVAILABLE:
        print(f"{colors.YELLOW}Google Generative AI SDK not installed. Skipping.{colors.ENDC}")
        return
    try:
        genai.configure(api_key=api_key)
        for model in models:
            print(f"Testing model: {model} ... ", end="")
            genai_model = genai.GenerativeModel(model)
            await genai_model.generate_content_async("This is a test.", generation_config=genai.types.GenerationConfig(
                max_output_tokens=5))
            print(f"{colors.GREEN}SUCCESS{colors.ENDC}")
    except Exception as e:
        print(f"{colors.RED}FAILED{colors.ENDC}")
        print(f"  Error: {e.__class__.__name__}: {e}")


# --- Main Execution ---

async def main():
    print("Starting API key and model access verification...")

    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        await verify_openai(openai_key.split(',')[0], MODELS_TO_TEST["openai"])
    else:
        print(f"\n{colors.YELLOW}OPENAI_API_KEY not found in .env file. Skipping.{colors.ENDC}")

    # Check Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        await verify_anthropic(anthropic_key.split(',')[0], MODELS_TO_TEST["anthropic"])
    else:
        print(f"\n{colors.YELLOW}ANTHROPIC_API_KEY not found in .env file. Skipping.{colors.ENDC}")

    # Check Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        await verify_gemini(gemini_key.split(',')[0], MODELS_TO_TEST["google_gemini"])
    else:
        print(f"\n{colors.YELLOW}GEMINI_API_KEY not found in .env file. Skipping.{colors.ENDC}")

    print("\nVerification complete.")


if __name__ == "__main__":
    asyncio.run(main())