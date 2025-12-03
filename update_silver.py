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
    TSMixer(h=horizon, n_series=1, input_size=504, n_block=4, ff_dim=128, dropout=0.5,
            revin=True, max_steps=350, learning_rate=3e-4, scaler_type="robust", batch_size=32),
    NBEATS(h=horizon, input_size=252, max_steps=350, learning_rate=5e-4, scaler_type="robust",
           n_blocks=[3, 3, 3], mlp_units=[[512, 512], [512, 512], [512, 512]],
           stack_types=["trend", "seasonality", "seasonality"], batch_size=32),
    NHITS(h=horizon, input_size=504, max_steps=350, learning_rate=5e-4, scaler_type="robust",
          n_freq_downsample=[20, 5, 1], interpolation_mode="cubic", pooling_mode="MaxPool1d",
          n_pool_kernel_size=[3, 3, 3], batch_size=32),
    MLP(h=horizon, input_size=126, max_steps=350, learning_rate=1e-3, scaler_type="robust",
        num_layers=2, hidden_size=256, batch_size=64),
    TiDE(h=horizon, input_size=504, max_steps=350, learning_rate=3e-4, scaler_type="robust",
         hidden_size=256, num_encoder_layers=2, num_decoder_layers=2, decoder_output_dim=16,
         temporal_width=4, batch_size=32, dropout=0.3),
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

