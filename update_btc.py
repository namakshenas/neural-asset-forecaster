from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixer, NBEATS, NHITS, MLP, TiDE
import yfinance as yf
import logging
import plotly.graph_objects as go

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

btc_df = yf.download("BTC-USD", start="2018-01-01")["Close"].astype(int).reset_index()
btc_df.columns = ["ds", "y"]
btc_df.insert(0, "unique_id", "1.0")

horizon = 30

models = [
    TSMixer(h=horizon, n_series=1, input_size=336, n_block=3, ff_dim=128, dropout=0.3, revin=True,
        scaler_type="identity", max_steps=400, learning_rate=5e-4, batch_size=32,early_stop_patience_steps=10),
    NBEATS(h=horizon, input_size=336, max_steps=350, learning_rate=1e-3, scaler_type="robust",
        n_blocks=[3, 3, 2], mlp_units=[[512, 512], [512, 512], [512, 512]], 
        stack_types=["trend", "seasonality", "identity"], batch_size=32),
    NHITS(h=horizon, input_size=336, max_steps=350, learning_rate=1e-3, scaler_type="robust",
        n_freq_downsample=[8, 4, 1], interpolation_mode="linear", 
        pooling_mode="MaxPool1d", activation="ReLU", batch_size=32),
    MLP(h=horizon, input_size=336, max_steps=350, learning_rate=5e-4, scaler_type="robust",
        num_layers=3, hidden_size=512, batch_size=32),
    TiDE(h=horizon, input_size=512, max_steps=350, learning_rate=5e-4, scaler_type="robust",
        hidden_size=256, batch_size=32),
]

print("Training models...")
nf = NeuralForecast(models=models, freq="D")
nf.fit(df=btc_df, val_size=horizon)

print("Generating predictions...")
Y_hat_df = nf.predict()

print("Generating figure...")
fig = go.Figure()
recent_data = btc_df.tail(60)
fig.add_trace(go.Scatter(x=recent_data["ds"], y=recent_data["y"], 
                        name="Actual", line=dict(color="black", width=3)))
for model in [type(model).__name__ for model in models]:
    fig.add_trace(go.Scatter(x=Y_hat_df["ds"], y=Y_hat_df[model], 
                            name=model, line=dict(width=2)))

fig.update_layout(title="Bitcoin Price Prediction - All Models", 
                xaxis_title="Date", yaxis_title="Price (USD)",
                hovermode="x unified", template="plotly_white")

fig.write_image("predictions/btc.png", width=1400, height=800)

fig.show()

