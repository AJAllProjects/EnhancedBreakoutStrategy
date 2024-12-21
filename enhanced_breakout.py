import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

############################
#   Technical Indicators   #
############################

def compute_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = rsi
    return df

def compute_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    df['EMA_fast'] = df['Close'].ewm(span=fastperiod, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slowperiod, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signalperiod, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

def compute_daily_vwap(df):
    """
    Computes a 5-day rolling VWAP for daily data using typical price.
    If the DataFrame has multi-level columns, we flatten them.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns.values]
        rename_map = {
            "Open_": "Open",
            "High_": "High",
            "Low_": "Low",
            "Close_": "Close",
            "Adj Close_": "Adj Close",
            "Volume_": "Volume"
        }
        df.rename(columns=rename_map, inplace=True)

    df["TypicalPrice"] = (df["High"] + df["Low"] + df["Close"]) / 3

    rolling_days = 5
    rolling_sum_pricevol = (df["TypicalPrice"] * df["Volume"]).rolling(rolling_days, min_periods=1).sum()
    rolling_sum_vol = df["Volume"].rolling(rolling_days, min_periods=1).sum()

    if isinstance(rolling_sum_pricevol, pd.DataFrame):
        rolling_sum_pricevol = rolling_sum_pricevol.squeeze(axis=1)
    if isinstance(rolling_sum_vol, pd.DataFrame):
        rolling_sum_vol = rolling_sum_vol.squeeze(axis=1)

    df["VWAP"] = rolling_sum_pricevol / rolling_sum_vol
    return df

#####################
#   Breakout Logic  #
#####################

def mark_breakouts(
        df,
        vol_threshold=200,
        price_threshold=2.0,
        rolling_window=20,
        consecutive_days=1
):
    df['AvgVolume'] = df['Volume'].rolling(rolling_window, min_periods=1).mean()

    df['Volume'], df['AvgVolume'] = df['Volume'].align(df['AvgVolume'], axis=0, copy=False)

    df['VolumeBreakoutSingle'] = df['Volume'] > (vol_threshold / 100.0) * df['AvgVolume']

    if consecutive_days > 1:
        df['VolConsecutive'] = (
            df['VolumeBreakoutSingle'].rolling(window=consecutive_days, min_periods=1).sum()
        )
        df['VolumeBreakout'] = df['VolConsecutive'] >= consecutive_days
    else:
        df['VolumeBreakout'] = df['VolumeBreakoutSingle']

    df['BarPctChange'] = df['Close'].pct_change() * 100
    df['PriceBreakout'] = df['BarPctChange'] > price_threshold

    df['IsBreakout'] = df['VolumeBreakout'] & df['PriceBreakout']

    df.fillna(False, inplace=True)
    return df

def apply_rsi_filter(df, rsi_min=None, rsi_max=None):
    if rsi_min is not None:
        df['IsBreakout'] = df['IsBreakout'] & (df['RSI'] >= rsi_min)
    if rsi_max is not None:
        df['IsBreakout'] = df['IsBreakout'] & (df['RSI'] <= rsi_max)
    return df

def apply_macd_filter(df, use_macd):
    if use_macd:
        df['IsBreakout'] = df['IsBreakout'] & (df['MACD'] > df['MACD_signal'])
    return df

def apply_vwap_filter(df, use_vwap):
    if use_vwap:
        df['IsBreakout'] = df['IsBreakout'] & (df['Close'] > df['VWAP'])
    return df

######################
#   Earnings Filter  #
######################

def apply_earnings_filter(df, ticker, buffer_days=2):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.get_earnings_dates(limit=20)
        earnings_dates = cal.index.date

        df_dates = df.index.date
        mask = np.ones(len(df), dtype=bool)
        for ed in earnings_dates:
            lower = ed - timedelta(days=buffer_days)
            upper = ed + timedelta(days=buffer_days)
            in_range = (df_dates >= lower) & (df_dates <= upper)
            mask[in_range] = False

        df['IsBreakout'] = df['IsBreakout'] & mask
    except Exception as e:
        print("Earnings filter error:", e)

    return df

#########################
#   News / Event Range  #
#########################

def parse_date_ranges(text_input):
    date_ranges = []
    lines = text_input.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "to" in line:
            parts = line.split("to")
        elif "-" in line:
            parts = line.split("-")
        else:
            parts = [line, line]

        parts = [p.strip() for p in parts]
        try:
            start_str = parts[0]
            end_str = parts[1] if len(parts) > 1 else parts[0]
            start_date = pd.to_datetime(start_str).date()
            end_date = pd.to_datetime(end_str).date()

            if start_date > end_date:
                start_date, end_date = end_date, start_date
            date_ranges.append((start_date, end_date))
        except Exception as e:
            print(f"Could not parse line: '{line}'. Error: {e}")
    return date_ranges

def apply_news_event_range_filter(df, date_ranges):
    if not date_ranges:
        return df

    df_dates = df.index.date
    mask = np.ones(len(df), dtype=bool)

    for (start_dt, end_dt) in date_ranges:
        in_range = (df_dates >= start_dt) & (df_dates <= end_dt)
        mask[in_range] = False

    df['IsBreakout'] = df['IsBreakout'] & mask
    return df

###################
#   Trade Logic   #
###################

def simulate_trades(
    df,
    holding_period,
    stop_loss_pct,
    profit_target_pct,
    buy_next_day_open=False
):
    trades = []
    n = len(df)
    i = 0
    while i < n - 1:
        if df['IsBreakout'].iloc[i]:
            if buy_next_day_open:
                if i + 1 < n:
                    buy_idx = i + 1
                    buy_time = df.index[buy_idx]
                    buy_price = df['Open'].iloc[buy_idx]
                else:
                    break
            else:
                buy_idx = i
                buy_time = df.index[buy_idx]
                buy_price = df['Close'].iloc[buy_idx]

            entry_price = buy_price
            exit_price = np.nan
            exit_time = None
            max_exit_idx = buy_idx + holding_period

            j = buy_idx + 1
            while j <= max_exit_idx and j < n:
                current_price = df['Close'].iloc[j]
                pct_move = (current_price - entry_price) / entry_price * 100

                if pct_move >= profit_target_pct:
                    exit_time = df.index[j]
                    exit_price = current_price
                    break
                elif pct_move <= stop_loss_pct:
                    exit_time = df.index[j]
                    exit_price = current_price
                    break

                j += 1

            if pd.isna(exit_price):
                if j < n:
                    exit_time = df.index[j - 1]
                    exit_price = df['Close'].iloc[j - 1]
                else:
                    exit_time = df.index[-1]
                    exit_price = df['Close'].iloc[-1]

            pct_return = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'BreakoutIndex': i,
                'BuyTime': buy_time,
                'BuyPrice': buy_price,
                'SellTime': exit_time,
                'SellPrice': exit_price,
                'PctReturn': pct_return
            })

            i = j
        else:
            i += 1

    trades_df = pd.DataFrame(trades)
    return trades_df

def benchmark_performance(benchmark_ticker, start_date, end_date, interval):
    bench_df = yf.download(benchmark_ticker, start=start_date, end=end_date, interval=interval)

    if isinstance(bench_df.columns, pd.MultiIndex):
        if benchmark_ticker in bench_df.columns.levels[1]:
            bench_df = bench_df.xs(benchmark_ticker, axis=1, level=1, drop_level=True)
        else:
            bench_df.columns = ["_".join(col).strip() for col in bench_df.columns.values]

    bench_df.dropna(inplace=True)
    if len(bench_df) < 2:
        return np.nan, np.nan, np.nan

    buy_price = bench_df['Close'].iloc[0]
    sell_price = bench_df['Close'].iloc[-1]
    bench_return = (sell_price - buy_price) / buy_price * 100
    return buy_price, sell_price, bench_return

################
#   Plotting   #
################

def plot_price_breakouts(df, ticker, theme="plotly_white"):
    """Basic price chart with candlestick + breakout markers."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f"{ticker} Price"
        ),
        secondary_y=False
    )

    breakout_indices = df.index[df['IsBreakout'] == True]
    if len(breakout_indices) > 0:
        breakout_close = df.loc[breakout_indices, 'Close']
        fig.add_trace(
            go.Scatter(
                x=breakout_indices,
                y=breakout_close,
                mode='markers',
                marker=dict(size=8, color='red', symbol='triangle-up'),
                name='Breakouts'
            ),
            secondary_y=False
        )

    fig.update_layout(
        title=f"{ticker} with Breakout Signals",
        xaxis_title="Date/Time",
        yaxis_title="Price",
        template=theme,
        hovermode="x unified"
    )
    return fig

def plot_technical_indicators(df, ticker, theme="plotly_white"):
    """
    Creates a multi-row figure showing:
    - Row 1: MACD (line & signal) + histogram
    - Row 2: RSI
    - Row 3: VWAP line (with 'Close' as reference)
    """

    rows = 3
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=[f"MACD - {ticker}",
                                        f"RSI - {ticker}",
                                        f"VWAP - {ticker}"])

    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_signal'],
                mode='lines',
                name='MACD_signal',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
        if 'MACD_hist' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_hist'],
                    name='MACD_hist',
                    marker_color='gray',
                    opacity=0.5
                ),
                row=1, col=1
            )

    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        fig.add_hrect(
            y0=70, y1=70, line_width=1, line_dash="dot", line_color="red",
            row=2, col=1
        )
        fig.add_hrect(
            y0=30, y1=30, line_width=1, line_dash="dot", line_color="green",
            row=2, col=1
        )

    if 'VWAP' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close',
                line=dict(color='black')
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['VWAP'],
                mode='lines',
                name='VWAP',
                line=dict(color='magenta')
            ),
            row=3, col=1
        )

    fig.update_layout(
        template=theme,
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="MACD", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Price/VWAP", row=3, col=1)

    return fig

def plot_trade_returns_distribution(trades_df, theme="plotly_white"):
    if trades_df.empty:
        return go.Figure()
    fig = px.histogram(
        trades_df,
        x="PctReturn",
        nbins=20,
        title="Distribution of Trade Returns",
        labels={"PctReturn": "Trade Return (%)"},
        template=theme
    )
    return fig

##################
#  Streamlit UI  #
##################

def main():
    st.set_page_config(
        page_title="Advanced Breakout with News/Event Range Filter",
        layout="wide"
    )

    st.title("Advanced Breakout Strategy (News/Event Date Range)")

    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime(2020,1,1))
    end_date = st.sidebar.date_input("End Date", datetime.today())
    interval = st.sidebar.selectbox("Interval", ["1d", "15m", "1h", "30m"], index=0)

    vol_threshold = st.sidebar.number_input("Volume Breakout Threshold (%)", 1, 1000, 200)
    price_threshold = st.sidebar.number_input("Price Breakout Threshold (%)", 0.1, 100.0, 2.0)
    consecutive_days = st.sidebar.number_input("Consecutive Volume Days", 1, 5, 1)

    holding_period = st.sidebar.number_input("Holding Period (bars)", 1, 1000, 10)
    stop_loss_pct = st.sidebar.number_input("Stop Loss (%)", -100.0, 0.0, -5.0)
    profit_target_pct = st.sidebar.number_input("Profit Target (%)", 0.0, 500.0, 10.0)
    buy_next_day_open = st.sidebar.checkbox("Buy on Next Bar's Open?", value=False)

    st.sidebar.subheader("Technical Filters")
    use_rsi = st.sidebar.checkbox("Compute RSI?", value=False)
    if use_rsi:
        rsi_min = st.sidebar.number_input("Min RSI", 0, 100, 0)
        rsi_max = st.sidebar.number_input("Max RSI", 0, 100, 100)
    else:
        rsi_min = None
        rsi_max = None

    use_macd = st.sidebar.checkbox("Use MACD bullish crossover?", value=False)
    use_vwap = st.sidebar.checkbox("Use VWAP filter (Close > VWAP)?", value=False)

    st.sidebar.subheader("Earnings Filter")
    use_earnings_filter = st.sidebar.checkbox("Avoid Earnings +/- X days?", value=False)
    earnings_buffer = st.sidebar.number_input("Earnings Buffer Days", 0, 10, 2) if use_earnings_filter else 0

    st.sidebar.subheader("News / Event Date Ranges")
    st.sidebar.markdown(
        "Enter each range on a new line, e.g.\n```\n2023-07-01 to 2023-07-03\n2023-09-10 - 2023-09-15\n2023-12-25\n```\n"
        "Single date => that exact day is excluded.\n"
    )
    user_event_str = st.sidebar.text_area("Event Date Ranges (one per line)", "")

    date_ranges = parse_date_ranges(user_event_str)

    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="^GSPC")
    plot_theme = st.sidebar.selectbox("Plotly Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"], index=0)

    if st.sidebar.button("Run Strategy"):
        with st.spinner("Fetching data..."):
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.levels[1]:
                    df = df.xs(ticker, axis=1, level=1, drop_level=True)
                else:
                    df.columns = ["_".join(col).strip() for col in df.columns.values]

            if df.empty:
                st.error("No data downloaded. Check your parameters.")
                return

        with st.spinner("Computing indicators..."):
            if use_rsi:
                df = compute_rsi(df)
            if use_macd:
                df = compute_macd(df)
            if use_vwap:
                df = compute_daily_vwap(df)

        with st.spinner("Marking breakouts..."):
            df = mark_breakouts(
                df,
                vol_threshold=vol_threshold,
                price_threshold=price_threshold,
                rolling_window=20,
                consecutive_days=consecutive_days
            )

            if use_rsi:
                df = apply_rsi_filter(df, rsi_min=rsi_min, rsi_max=rsi_max)

            df = apply_macd_filter(df, use_macd)
            df = apply_vwap_filter(df, use_vwap)

            if use_earnings_filter:
                df = apply_earnings_filter(df, ticker, buffer_days=earnings_buffer)

            df = apply_news_event_range_filter(df, date_ranges)

        with st.spinner("Simulating trades..."):
            trades_df = simulate_trades(
                df,
                holding_period=holding_period,
                stop_loss_pct=stop_loss_pct,
                profit_target_pct=profit_target_pct,
                buy_next_day_open=buy_next_day_open
            )

        if trades_df.empty:
            st.warning("No trades triggered.")
        else:
            st.subheader("Trade Results")
            st.dataframe(trades_df, use_container_width=True)

            avg_return = trades_df['PctReturn'].mean()
            median_return = trades_df['PctReturn'].median()
            total_trades = len(trades_df)
            positive_trades = sum(trades_df['PctReturn'] > 0)
            win_rate = (positive_trades / total_trades) * 100 if total_trades > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", f"{total_trades}")
            col2.metric("Avg Return (%)", f"{avg_return:.2f}")
            col3.metric("Median Return (%)", f"{median_return:.2f}")
            col4.metric("Win Rate (%)", f"{win_rate:.2f}")

            fig_dist = plot_trade_returns_distribution(trades_df, theme=plot_theme)
            st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Price Chart with Breakouts")
        fig_price = plot_price_breakouts(df, ticker, theme=plot_theme)
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("Technical Indicators")
        fig_indicators = plot_technical_indicators(df, ticker, theme=plot_theme)
        st.plotly_chart(fig_indicators, use_container_width=True)

        bench_buy, bench_sell, bench_return = benchmark_performance(benchmark_ticker, start_date, end_date, interval)

        st.subheader("Benchmark Performance")
        if not np.isnan(bench_return):
            st.write(f"{benchmark_ticker} Buy Price: {bench_buy:.2f}")
            st.write(f"{benchmark_ticker} Sell Price: {bench_sell:.2f}")
            st.write(f"{benchmark_ticker} Return: {bench_return:.2f}%")
        else:
            st.warning("Could not fetch benchmark data or not enough data for benchmark.")

        st.subheader("Download Full Data & Trades")
        full_data_csv = df.to_csv(index=True)
        st.download_button(
            label="Download Full Data CSV",
            data=full_data_csv,
            file_name=f"{ticker}_full_data.csv",
            mime="text/csv"
        )

        if not trades_df.empty:
            csv_data = trades_df.to_csv(index=False)
            st.download_button(
                label="Download Trades CSV",
                data=csv_data,
                file_name=f"{ticker}_trades.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
