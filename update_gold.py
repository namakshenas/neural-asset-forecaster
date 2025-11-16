from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixer, NBEATS, NHITS, MLP, TiDE
import yfinance as yf
import logging
import plotly.graph_objects as go

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

gld_df = yf.download("GC=F", start="2018-01-01")["Close"].astype(int).reset_index()
gld_df.columns = ["ds", "y"]
gld_df.insert(0, "unique_id", "1.0")

horizon = 30

models = [
    TSMixer(h=horizon, n_series=1, input_size=512, n_block=5, ff_dim=256, dropout=0.35,
            revin=True, max_steps=350, learning_rate=1e-3, scaler_type="standard", batch_size=32),
    NBEATS(h=horizon, input_size=336, max_steps=350, learning_rate=1e-3, scaler_type="robust",
           n_blocks=[4, 4], mlp_units=[[512, 512], [512, 512]], stack_types=["trend", "seasonality"], batch_size=32),
    NHITS(h=horizon, input_size=512, max_steps=350, learning_rate=1e-3, scaler_type="robust",
          n_freq_downsample=[8, 4, 2, 1], interpolation_mode="linear", pooling_mode="MaxPool1d", activation="ReLU", batch_size=32),
    MLP(h=horizon, input_size=336, max_steps=350, learning_rate=1e-3, scaler_type="standard",
        num_layers=4, hidden_size=512, batch_size=32),
    TiDE(h=horizon, input_size=720, max_steps=350, learning_rate=1e-3, scaler_type="standard",
         hidden_size=512, decoder_output_dim=32, temporal_decoder_dim=128, dropout=0.3, batch_size=32),
]

print("Training models...")
nf = NeuralForecast(models=models, freq="D")
nf.fit(df=gld_df)

print("Generating predictions...")
Y_hat_df = nf.predict()

# Create plotly figure
fig = go.Figure()

# Add actual data
recent_data = gld_df.tail(60)
fig.add_trace(go.Scatter(x=recent_data["ds"], y=recent_data["y"], 
                         name="Actual", line=dict(color="black", width=3)))

# Add forecasts
models_list = ["TSMixer", "NBEATS", "NHITS", "MLP", "TiDE"]
for model in models_list:
    fig.add_trace(go.Scatter(x=Y_hat_df["ds"], y=Y_hat_df[model], 
                             name=model, line=dict(width=2)))

fig.update_layout(title="GOLD Price Prediction - All Models", 
                  xaxis_title="Date", yaxis_title="Price (USD)",
                  hovermode="x unified", template="plotly_white")

fig.write_image("predictions/gold.png", width=1400, height=800)
fig.show()

