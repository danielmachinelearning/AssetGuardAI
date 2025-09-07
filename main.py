import yfinance as yf
from pyteal import *
from algosdk.v2client import algod
from langgraph.graph import StateGraph
import feedparser
from anthropic import Anthropic

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# -------------------
# Algorand Setup
# -------------------
ALGOD_ADDRESS = "https://testnet-api.algonode.cloud"
ALGOD_TOKEN = ""
HEADERS = {}

# -------------------
# Anthropic Setup
# -------------------
ANTHROPIC_API_KEY = ""
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# -------------------
# News Node
# -------------------
def fetch_and_summarize_news(state: dict) -> dict:
    rss_feeds = [
        "https://www.reddit.com/r/algorand/new/.rss",
        "https://cointelegraph.com/rss/tag/algorand"
    ]
    news_list = []
    for rss_url in rss_feeds:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            news_list.append(f"{title} ({link})")

    if not news_list:
        return {**state, "news": [], "news_summary": "No news available.", "news_sentiment": 0.0}

    prompt = (
        "Summarize the following news about Algorand concisely and provide a sentiment score "
        "from -1 (very negative) to +1 (very positive), 0 for neutral. Return ONLY the numeric "
        "sentiment as a floating-point number along with the summary.\n\n" +
        "\n".join(news_list)
    )

    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=300,
            system="You summarize news and provide sentiment.",
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.content[0].text.strip()
        if "Sentiment:" in result_text:
            summary_part, sentiment_part = result_text.rsplit("Sentiment:", 1)
            summary = summary_part.strip()
            try:
                sentiment = float(sentiment_part.strip())
                sentiment = max(min(sentiment, 1.0), -1.0)
            except:
                sentiment = 0.0
        else:
            summary = result_text
            sentiment = 0.0

    except Exception as e:
        summary = "Failed to summarize news."
        sentiment = 0.0

    return {**state, "news": news_list, "news_summary": summary, "news_sentiment": sentiment}

# -------------------
# Forecast Node
# -------------------
def forecast_time_series_trend(ticker=None, n_forecast=7, price_col="Close") -> dict:
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        ticker = "ALGO-USD"

    data = yf.download(ticker, period="7d", interval="1h", auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data fetched for ticker '{ticker}'")
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found")

    close_series = TimeSeries.from_dataframe(data, value_cols=price_col)

    # Covariates
    hour_series = datetime_attribute_timeseries(close_series, "hour", one_hot=True)
    weekday_series = datetime_attribute_timeseries(close_series, "weekday", one_hot=True)
    future_covariates = hour_series.stack(weekday_series)

    # Ensure covariates cover forecast horizon
    last_time = close_series.end_time()
    forecast_end_time = last_time + pd.Timedelta(hours=n_forecast)
    if future_covariates.end_time() < forecast_end_time:
        all_times = pd.date_range(start=close_series.start_time(), end=forecast_end_time, freq="h")
        dummy_series = TimeSeries.from_times_and_values(all_times, np.zeros(len(all_times)))
        hour_series = datetime_attribute_timeseries(dummy_series, "hour", one_hot=True)
        weekday_series = datetime_attribute_timeseries(dummy_series, "weekday", one_hot=True)
        future_covariates = hour_series.stack(weekday_series)

    # Train TFT
    model = TFTModel(
        input_chunk_length=24,
        output_chunk_length=n_forecast,
        hidden_size=16,
        lstm_layers=1,
        batch_size=16,
        n_epochs=10,
        dropout=0.1,
        random_state=42,
        log_tensorboard=False
    )
    model.fit(close_series, future_covariates=future_covariates, verbose=True)
    forecast_series = model.predict(n=n_forecast, future_covariates=future_covariates)

    values = forecast_series.values()[:, 0]
    trend = "neutral"
    if values[-1] > values[0]:
        trend = "uptrend"
    elif values[-1] < values[0]:
        trend = "downtrend"

    pct_change = ((values[-1] - values[0]) / values[0]) * 100

    return {"trend": trend, "forecast_series": forecast_series, "trend_pct": pct_change}

# -------------------
# Smart Contract Node
# -------------------
def create_price_contract(state: dict, threshold_pct: float, target_price: float, action: str) -> dict:
    """
    action: "buy" or "sell"
    Only create contract if trend_pct exceeds threshold
    """
    trend_pct = state.get("trend_pct", 0.0)
    trend = state.get("trend", "neutral")

    if abs(trend_pct) < threshold_pct:
        print(f"[INFO] Trend pct {trend_pct:.2f}% below threshold {threshold_pct}%, contract not created")
        return {**state, "contract_code": None, "contract_created": False}

    if action.lower() == "buy":
        contract_logic = App.globalPut(Bytes("buy_price"), Int(target_price))
        contract = Seq([contract_logic, Approve()])
    elif action.lower() == "sell":
        contract_logic = App.globalPut(Bytes("sell_price"), Int(target_price))
        contract = Seq([contract_logic, Approve()])
    else:
        raise ValueError("Action must be 'buy' or 'sell'")

    contract_code = compileTeal(contract, mode=Mode.Signature, version=5)
    print(f"[INFO] Contract created for {action} at price {target_price} with trend_pct {trend_pct:.2f}%")
    return {**state, "contract_code": contract_code, "contract_created": True}

# -------------------
# Smart Contract Node (manual inputs)
# -------------------
def create_price_contract_manual(state: dict) -> dict:
    print("=== Create Smart Contract ===")
    action = input("Enter action (buy/sell): ").lower().strip()
    pct_threshold = float(input("Enter trend percentage threshold (e.g., 1.5 for 1.5%): "))
    target_price = float(input("Enter target price: "))

    # Convert float to integer micro-units (e.g., 34.5 -> 34500000)
    price_micro = int(target_price * 1_000_000)
    threshold_micro = int(pct_threshold * 1_000_000)

    # Simple approval logic: always approve (manual contract)
    contract_logic = Seq([
        App.globalPut(Bytes("action"), Bytes(action)),
        App.globalPut(Bytes("buy_sell_price"), Int(price_micro)),
        App.globalPut(Bytes("trend_threshold"), Int(threshold_micro)),
        Approve()
    ])

    contract_code = compileTeal(contract_logic, mode=Mode.Application, version=5)
    print(f"[INFO] Manual contract created: action={action}, price={target_price}, threshold={pct_threshold}%")
    return {**state, "contract_code": contract_code}


# -------------------
# Deploy Contract Node
# -------------------
def deploy_contract(state: dict) -> dict:
    if not state.get("contract_code"):
        return {**state, "contract_hash": None, "contract_deployed": False}

    client = algod.AlgodClient(ALGOD_TOKEN, ALGOD_ADDRESS, headers=HEADERS)
    response = client.compile(state["contract_code"])
    print("[INFO] Contract compiled on-chain, hash:", response.get("hash"))
    return {**state, "contract_hash": response.get("hash"), "contract_deployed": True}

# -------------------
# Output Node
# -------------------
def output_results(state: dict) -> dict:
    print("\n==== RESULTS ====")
    for k, v in state.items():
        print(f"{k}: {v}")
    print("================\n")
    return state

# -------------------
# LangGraph Workflows
# -------------------

# News Workflow
news_workflow = StateGraph(dict)
news_workflow.add_node("fetch_news", fetch_and_summarize_news)
news_workflow.add_node("output_results", output_results)
news_workflow.add_edge("fetch_news", "output_results")
news_workflow.set_entry_point("fetch_news")
news_workflow.set_finish_point("output_results")
news_app = news_workflow.compile()

# Price/Forecast Workflow
price_workflow = StateGraph(dict)
price_workflow.add_node("fetch_time_series", forecast_time_series_trend)
price_workflow.add_node("output_results", output_results)
price_workflow.add_edge("fetch_time_series", "output_results")
price_workflow.set_entry_point("fetch_time_series")
price_workflow.set_finish_point("output_results")
price_app = price_workflow.compile()

# Smart Contract Workflow
contract_workflow = StateGraph(dict)
contract_workflow.add_node("create_contract", create_price_contract_manual)
contract_workflow.add_node("deploy_contract", deploy_contract)
contract_workflow.add_node("output_results", output_results)
contract_workflow.add_edge("create_contract", "deploy_contract")
contract_workflow.add_edge("deploy_contract", "output_results")
contract_workflow.set_entry_point("create_contract")
contract_workflow.set_finish_point("output_results")
contract_app = contract_workflow.compile()

# -------------------
# Dispatcher
# -------------------
user_query = input("Enter your query: ").lower()
initial_state = {}

if "news" in user_query:
    final_state = news_app.invoke(initial_state)
elif "time series" in user_query or "price" in user_query or "trend" in user_query:
    final_state = price_app.invoke(initial_state)
elif "smart contract" in user_query:
    final_state = contract_app.invoke(initial_state)
else:
    print("[INFO] Defaulting to time series")
    final_state = price_app.invoke(initial_state)