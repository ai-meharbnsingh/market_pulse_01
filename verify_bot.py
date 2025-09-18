# verify_bot.py

import os
import requests
from dotenv import load_dotenv


def send_telegram_message(message):
    """Sends a message to the Telegram chat defined in the .env file."""

    # Load environment variables from .env file
    load_dotenv()

    # Get bot token and chat ID from environment variables
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    # --- Basic Validation ---
    if not BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found. Make sure it's set in your .env file.")
        return
    if not CHAT_ID:
        print("Error: TELEGRAM_CHAT_ID not found. Make sure it's set in your .env file.")
        return

    # Construct the API URL
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # Prepare the data payload
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'  # Optional: for formatting like bold, italic, etc.
    }

    print("Sending message to Telegram...")

    try:
        # Make the POST request to the Telegram API
        response = requests.post(url, data=payload)

        # Convert the response to JSON to check its contents
        response_json = response.json()

        # Check if the message was sent successfully
        if response.status_code == 200 and response_json['ok']:
            print("✅ Message sent successfully!")
        else:
            print("❌ Failed to send message.")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response_json}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the network request: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    test_message = "Hello from your Python script! 👋\nIf you see this, your bot is working correctly."
    send_telegram_message(test_message)