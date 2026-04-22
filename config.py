DATA_SYMBOL = "^NSEI"

INTERVAL = "5m"
PERIOD = "60d"

SEQ_LENGTH = 60

HMM_FEATURES = [
    "log_return",
    "volatility",
    "rsi",
    "macd"
]

FEATURE_COLUMNS = [
"log_return",
"volatility",
"rsi",
"macd",
"price_range",
"momentum",
"trend",
"trend_strength",
"volatility_break",
"range_strength",
"volume_pressure",
"volume_volatility",
"turnover",
"rel_volume"
]

HMM_ITERATIONS = 500