from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixer, NBEATS, NHITS, MLP, PatchTST, TiDE
import yfinance as yf
import re
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Download BTC data
btc_df = yf.download("BTC-USD", start="2018-01-01")["Close"].astype(int).reset_index()
btc_df.columns = ["ds", "y"]
btc_df.insert(0, "unique_id", "1.0")

horizon = 30

models = [
    TSMixer(h=horizon, n_series=1, input_size=336, n_block=4, ff_dim=128, dropout=0.3, 
            revin=True, max_steps=400, learning_rate=1e-3, scaler_type="robust", batch_size=32),
    # NBEATS(h=horizon, input_size=168, max_steps=500, learning_rate=1e-3, scaler_type="robust",
    #        n_blocks=[3, 3], mlp_units=[[256, 256], [256, 256]], stack_types=["trend", "seasonality"], batch_size=32),
    # NHITS(h=horizon, input_size=336, max_steps=500, learning_rate=1e-3, scaler_type="robust",
    #       n_freq_downsample=[8, 4, 1], interpolation_mode="linear", pooling_mode="MaxPool1d", activation="ReLU", batch_size=32),
    # MLP(h=horizon, input_size=168, max_steps=400, learning_rate=1e-3, scaler_type="robust",
    #     num_layers=3, hidden_size=256, batch_size=32),
    # PatchTST(h=horizon, input_size=336, max_steps=500, learning_rate=2e-4, scaler_type="robust",
    #          patch_len=16, stride=8, encoder_layers=3, n_heads=16, revin=True, batch_size=32),
    # TiDE(h=horizon, input_size=720, max_steps=500, learning_rate=1e-3, scaler_type="robust",
    #      hidden_size=256, batch_size=32),
]

print("Training models...")
nf = NeuralForecast(models=models, freq="D")
nf.fit(df=btc_df)

print("Generating predictions...")
Y_hat_df = nf.predict()

# Generate Mermaid chart
recent_data = btc_df.tail(30)
dates = [d.strftime("%m/%d") for d in recent_data["ds"]] + [d.strftime("%m/%d") for d in Y_hat_df["ds"]]
actual = [str(int(v)) for v in recent_data["y"]]
last_price = int(recent_data["y"].iloc[-1])

forecast = [str(last_price)] + [str(int(v)) for v in Y_hat_df["TSMixer"][1:]]

mermaid = f"""```mermaid
xychart-beta
  title "Bitcoin Price - 30 Day Forecast"
  x-axis [{", ".join([f'"{d}"' for d in dates])}]
  y-axis "Price (USD)"
  line "Actual" [{", ".join(actual + ["0"] * len(Y_hat_df))}]
  line "Forecast" [{", ".join(["0"] * len(recent_data) + forecast)}]
```"""

# Update README
with open("README.md", "r") as f:
    readme = f.read()

readme = re.sub(r"<!-- BTC-START -->.*?<!-- BTC-END -->", 
                f"<!-- BTC-START -->\n\n{mermaid}\n\n<!-- BTC-END -->", 
                readme, flags=re.DOTALL)

with open("README.md", "w") as f:
    f.write(readme)

print("README updated!")

#   line "NBEATS" [{", ".join([""] * len(recent_data) + [str(int(v)) for v in Y_hat_df["NBEATS"]])}]
#   line "NHITS" [{", ".join([""] * len(recent_data) + [str(int(v)) for v in Y_hat_df["NHITS"]])}]
#   line "MLP" [{", ".join([""] * len(recent_data) + [str(int(v)) for v in Y_hat_df["MLP"]])}]
#   line "PatchTST" [{", ".join([""] * len(recent_data) + [str(int(v)) for v in Y_hat_df["PatchTST"]])}]
#   line "TiDE" [{", ".join([""] * len(recent_data) + [str(int(v)) for v in Y_hat_df["TiDE"]])}]

