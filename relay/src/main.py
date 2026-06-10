import os
import asyncio
import hashlib
import json
from telethon import TelegramClient, events
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHANNEL_MAPPING_JSON = os.getenv('CHANNEL_MAPPING') # JSON format: {"staging_id_1": {"target": "prod_id_1", "type": "signal"}, ...}

if not all([API_ID, API_HASH, BOT_TOKEN, CHANNEL_MAPPING_JSON]):
    logger.error("Missing required environment variables. Please check your .env file.")
    exit(1)

try:
    API_ID = int(API_ID)
except ValueError:
    logger.error("API_ID must be an integer.")
    exit(1)

# Parse channel mapping
try:
    mapping_raw = json.loads(CHANNEL_MAPPING_JSON)
    CHANNEL_MAPPING = {}
    for staging, config in mapping_raw.items():
        staging_id = int(staging) if str(staging).lstrip('-').isdigit() else staging
        target_id = config.get("target")
        if target_id is not None:
             target_id = int(target_id) if str(target_id).lstrip('-').isdigit() else target_id

        CHANNEL_MAPPING[staging_id] = {
            "target": target_id,
            "type": config.get("type", "default")
        }

    STAGING_CHANNELS = list(CHANNEL_MAPPING.keys())

except json.JSONDecodeError:
    logger.error("CHANNEL_MAPPING must be a valid JSON string.")
    exit(1)
except AttributeError:
     logger.error("CHANNEL_MAPPING values must be objects containing at least a 'target'.")
     exit(1)

DISCLAIMER_TEXT = "\n\nAI-generated content may contain errors. DYOR and verify relevant information. No financial advice. You are solely responsible for your trades and actions."

# Initialize the client
client = TelegramClient('relay_bot_session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)

# Dictionary to store message hashes and their timestamps for deduplication
message_cache = {}
CACHE_TIMEOUT = 5 * 60 # 5 minutes in seconds

async def clear_cache_periodically():
    while True:
        await asyncio.sleep(60)
        current_time = asyncio.get_event_loop().time()
        keys_to_delete = []
        for h, timestamp in message_cache.items():
            if current_time - timestamp > CACHE_TIMEOUT:
                keys_to_delete.append(h)
        for h in keys_to_delete:
            del message_cache[h]

def get_message_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def process_message(text, channel_type):
    """
    Apply different processing logic based on the channel type.
    """
    processed_text = text

    # We can add different logic per channel type here later.
    # For now, all channels just get the disclaimer.
    if channel_type == "signal":
        # Example placeholder for future logic
        pass
    elif channel_type == "market_analysis":
        # Example placeholder for future logic
        pass

    # Apply global logic (disclaimer)
    processed_text += DISCLAIMER_TEXT

    return processed_text

@client.on(events.NewMessage(chats=STAGING_CHANNELS))
async def handler(event):
    staging_chat_id = event.chat_id
    logger.info(f"Received message from chat_id: {staging_chat_id}")

    if staging_chat_id not in CHANNEL_MAPPING:
        logger.warning(f"Message from unmapped channel {staging_chat_id}, skipping.")
        return

    logger.info(f"Message is from a mapped staging channel: {staging_chat_id}")

    config = CHANNEL_MAPPING[staging_chat_id]
    production_channel = config["target"]
    channel_type = config["type"]

    if not event.message.message:
        logger.info(f"Message from {staging_chat_id} has no text, skipping.")
        return

    text = event.message.message
    msg_hash = get_message_hash(text)
    current_time = asyncio.get_event_loop().time()

    logger.info(f"Message hash generated: {msg_hash}")

    if msg_hash in message_cache:
        # Check if it's within the timeout window
        if current_time - message_cache[msg_hash] <= CACHE_TIMEOUT:
            logger.info(f"Duplicate message detected (hash: {msg_hash}), skipping.")
            return

    # Update cache
    message_cache[msg_hash] = current_time
    logger.info(f"Message added to cache (hash: {msg_hash})")

    # Process based on type
    processed_text = process_message(text, channel_type)

    try:
        await client.send_message(production_channel, processed_text)
        logger.info(f"Message successfully forwarded from {staging_chat_id} to production channel {production_channel}.")
    except Exception as e:
        logger.error(f"Failed to send message to {production_channel}: {e}")

async def main():
    logger.info("Starting Relay Bot...")
    logger.info(f"Loaded CHANNEL_MAPPING: {CHANNEL_MAPPING}")
    logger.info(f"Monitoring STAGING_CHANNELS: {STAGING_CHANNELS}")
    asyncio.create_task(clear_cache_periodically())
    await client.run_until_disconnected()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
