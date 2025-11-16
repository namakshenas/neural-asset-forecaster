from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixer, NBEATS, NHITS, MLP, TiDE
import yfinance as yf
import logging
import plotly.graph_objects as go

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

slv_df = yf.download("SI=F", start="2018-01-01")["Close"].astype(int).reset_index()
slv_df.columns = ["ds", "y"]
slv_df.insert(0, "unique_id", "1.0")

horizon = 30

models = [
    TSMixer(h=horizon, n_series=1, input_size=504, n_block=6, ff_dim=256, dropout=0.4,
            revin=True, max_steps=350, learning_rate=5e-4, scaler_type="robust", batch_size=24),
    NBEATS(h=horizon, input_size=336, max_steps=350, learning_rate=8e-4, scaler_type="robust",
           n_blocks=[4, 4], mlp_units=[[512, 512], [512, 512]], 
           stack_types=["trend", "seasonality"], batch_size=24),
    NHITS(h=horizon, input_size=720, max_steps=350, learning_rate=7e-4, scaler_type="robust",
          n_freq_downsample=[12, 6, 1], interpolation_mode="linear", pooling_mode="MaxPool1d", 
          n_pool_kernel_size=[12, 6, 1], batch_size=16),
    MLP(h=horizon, input_size=252, max_steps=300, learning_rate=8e-4, scaler_type="robust",
        num_layers=4, hidden_size=512, batch_size=24),
    TiDE(h=horizon, input_size=1008, max_steps=350, learning_rate=6e-4, scaler_type="robust",
         hidden_size=512, num_encoder_layers=3, num_decoder_layers=2, decoder_output_dim=32,
         temporal_width=8, batch_size=16),
]

print("Training models...")
nf = NeuralForecast(models=models, freq="D")
nf.fit(df=slv_df)

print("Generating predictions...")
Y_hat_df = nf.predict()

# Create plotly figure
fig = go.Figure()

# Add actual data
recent_data = slv_df.tail(60)
fig.add_trace(go.Scatter(x=recent_data["ds"], y=recent_data["y"], 
                         name="Actual", line=dict(color="black", width=3)))

# Add forecasts
models_list = ["TSMixer", "NBEATS", "NHITS", "MLP", "TiDE"]
for model in models_list:
    fig.add_trace(go.Scatter(x=Y_hat_df["ds"], y=Y_hat_df[model], 
                             name=model, line=dict(width=2)))

fig.update_layout(title="SILVER Price Prediction - All Models", 
                  xaxis_title="Date", yaxis_title="Price (USD)",
                  hovermode="x unified", template="plotly_white")

fig.write_image("predictions/silver.png", width=1400, height=800)
fig.show()

