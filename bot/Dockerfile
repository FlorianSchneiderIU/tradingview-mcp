FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached)
COPY bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source and the research/runtime modules it imports.
COPY bot/bot.py bot/indicators.py bot/train_dt.py bot/turtle_soup.py bot/session_orb.py bot/wolfe_wave.py bot/wolfe_dca_bot.py bot/ggshot_227.py bot/fill_notifier.py bot/matrix_signal_bot.py bot/rl_execution_bot.py bot/weekday_edge_bot.py bot/capitulation_spring_bot.py bot/capitulation_spring.py bot/market_context.py ./
COPY bot/configs ./configs
COPY scripts/backtest_turtle_soup.py scripts/backtest_wolfe_wave.py scripts/backtest_ggshot_227.py scripts/experiment_pine_strategy_candidates.py scripts/ml_trade_outcome_filter.py scripts/crypto_symbol_sets.py scripts/plot_zone_channel_history.py scripts/experiment_session_orb.py scripts/experiment_session_orb_fast.py scripts/evaluate_rl_split_head_replay.py scripts/rl_dashboard.py ./scripts/
COPY scripts/channel_state_research ./scripts/channel_state_research

# Create logs dir
RUN mkdir -p /app/logs

# Unbuffered stdout so logs appear immediately in `docker compose logs -f`
ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
