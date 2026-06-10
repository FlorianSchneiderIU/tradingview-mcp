# Relay Bot

This bot relays messages from staging channels to a production channel.
It deduplicates messages within a 5-minute window and appends a disclaimer.

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials.
2. Ensure you have the staging and production channel IDs.
3. Use Docker Compose to build and run the bot.

## Docker

The bot is designed to run via Docker Compose, alongside other services.

```bash
docker-compose up -d relay-bot
```
