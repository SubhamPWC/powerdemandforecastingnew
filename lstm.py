import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import io

st.set_page_config(layout="wide")
st.title("🔮 LSTM-Based Power Demand Forecast")

# 📤 Upload excel
uploaded_file = st.file_uploader("Upload excel File", type=["xlsx"])

# 🔧 Model Hyperparameters
st.sidebar.header("⚙️ Model Settings")
epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=10, step=5)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # 🕒 Parse datetime
    df['date'] = pd.to_datetime(df['date'])
	df['hour'] = pd.to_datetime(df['time'],format='%H:%M:%S').dt.hour
	df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')

    # 🧠 Feature engineering
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 10 or 17 <= x <= 20 else 0)
    df['weekend tag'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['holiday tag'] = 0

    # 🔠 Encode state
    le = LabelEncoder()
    df['state_encoded'] = le.fit_transform(df['state'])

    # ⏳ Lag features
    df['demand_lag1'] = df.groupby('state')['demand'].shift(1)
    df['demand_lag24'] = df.groupby('state')['demand'].shift(24)
    df['demand_rolling_mean_24'] = df.groupby('state')['demand'].rolling(window=24).mean().reset_index(level=0, drop=True)

    df = df.dropna()

    # 🎯 Features and target
    features = ['temperature', 'rain', 'DNI', 'weekend tag', 'holiday tag',
                'hour', 'dayofweek', 'month', 'is_peak_hour',
                'state_encoded', 'demand_lag1', 'demand_lag24', 'demand_rolling_mean_24']
    target = 'demand'

    # 📊 Normalize
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    df[target] = scaler.fit_transform(df[[target]])

    # 🧱 Sequence creation
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, -1])
        return np.array(X), np.array(y)

    seq_length = 24
    data = df[features + [target]].values
    X, y = create_sequences(data, seq_length)

    # 🧠 LSTM model
    model = Sequential([
        LSTM(64, input_shape=(seq_length, len(features)+1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    history = model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)

    # 📈 Historical predictions
    df['forecasted_demand'] = np.nan
    for i in range(seq_length, len(df)):
        seq = df[features + [target]].iloc[i-seq_length:i].values
        seq = np.expand_dims(seq, axis=0)
        df.loc[df.index[i], 'forecasted_demand'] = model.predict(seq, verbose=0)[0][0]

    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
    historical_df = df[['state', 'date', 'time', 'demand', 'forecasted_demand'] + features].copy()
    historical_df = historical_df.rename(columns={
        'demand': 'Actual Demand',
        'forecasted_demand': 'Forecasted Demand'
    })
    historical_df['Type'] = 'Historical'

    # 🔮 Future forecast
    future_forecasts = []
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        last_sequence = state_df[features + ['forecasted_demand']].tail(seq_length).values

        for i in range(1200):  # ~50 days hourly
            input_seq = np.expand_dims(last_sequence, axis=0)
            next_pred = model.predict(input_seq, verbose=0)[0][0]

            next_dt = state_df.iloc[-1]['datetime'] + pd.Timedelta(hours=1)
            hour = next_dt.hour
            dayofweek = next_dt.dayofweek
            month = next_dt.month
            is_peak_hour = 1 if 7 <= hour <= 10 or 17 <= hour <= 20 else 0
            weekend_tag = 1 if dayofweek >= 5 else 0
            holiday_tag = 0

            temp = state_df.iloc[-1]['temperature']
            rain = state_df.iloc[-1]['rain']
            dni = state_df.iloc[-1]['DNI']
            state_encoded = state_df.iloc[-1]['state_encoded']

            lag1 = last_sequence[-1][-1]
            lag24 = last_sequence[-24][-1] if len(last_sequence) >= 24 else lag1
            rolling_mean_24 = last_sequence[-24:][:, -1].mean()

            next_row = np.array([
                temp, rain, dni, weekend_tag, holiday_tag,
                hour, dayofweek, month, is_peak_hour,
                state_encoded, lag1, lag24, rolling_mean_24, next_pred
            ])
            last_sequence = np.vstack([last_sequence[1:], next_row])

            future_forecasts.append({
                'state': state,
                'date': next_dt.date(),
                'time': next_dt.strftime('%H:%M:%S'),
                'Actual Demand': np.nan,
                'Forecasted Demand': next_pred,
                'Type': 'Forecast'
            })

    future_df = pd.DataFrame(future_forecasts)
    combined_df = pd.concat([historical_df[['state', 'date', 'time', 'Actual Demand', 'Forecasted Demand', 'Type']], future_df], ignore_index=True)

    # 🔍 Filters
    st.sidebar.header("🔍 Data Filters")
    selected_state = st.sidebar.selectbox("Select State", combined_df['state'].unique())
    date_range = st.sidebar.date_input("Select Date Range", [combined_df['date'].min(), combined_df['date'].max()])

    filtered_df = combined_df[
        (combined_df['state'] == selected_state) &
        (combined_df['date'] >= date_range[0]) &
        (combined_df['date'] <= date_range[1])
    ]

    # 📈 Plot
    st.subheader(f"📈 Actual vs Forecasted Demand for {selected_state}")
    filtered_df['datetime'] = pd.to_datetime(filtered_df['date'].astype(str) + ' ' + filtered_df['time'])
    plot_df = filtered_df.set_index('datetime')[['Actual Demand', 'Forecasted Demand']]
    st.line_chart(plot_df)

    # 📊 Model Report
    st.sidebar.subheader("📊 Model Report")
    st.sidebar.write(f"Training Loss (Final Epoch): {history.history['loss'][-1]:.4f}")
    st.sidebar.write(f"Validation Loss (Final Epoch): {history.history['val_loss'][-1]:.4f}")

    # 📁 Export CSV
    st.subheader("📁 Export Combined Forecast CSV")
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_buffer.getvalue(),
        file_name="combined_actual_and_forecasted_demand_lstm.csv",
        mime="text/csv"
    )