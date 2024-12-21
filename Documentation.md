# Advanced Breakout Strategy — **Documentation**

This document provides an **in-depth explanation** of the **Advanced Breakout Strategy** project, describing the logic behind each component and why certain design choices were made. The code implements a multi-filter approach to detect potential breakout trades in a stock or index, and it uses **Streamlit** to present a user-friendly interface.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [File Structure](#file-structure)  
3. [Main Code Flow](#main-code-flow)  
4. [Key Modules & Functions](#key-modules--functions)  
   - [Technical Indicator Calculations](#technical-indicator-calculations)  
   - [Breakout Logic](#breakout-logic)  
   - [Filters](#filters)  
   - [Trade Simulation](#trade-simulation)  
   - [Benchmark](#benchmark)  
5. [Streamlit App Layout](#streamlit-app-layout)  
6. [Data Handling & Caveats](#data-handling--caveats)  
7. [Extending the Code](#extending-the-code)

---

## Introduction

The **Advanced Breakout Strategy** application is designed to detect high-potential breakout signals in a given ticker (e.g., stocks, ETFs) using a combination of **volume spikes**, **price changes**, and **advanced filters** (RSI, MACD, VWAP, earnings, and user-defined date ranges). 

### **Key Features**:
- **Multiple-Day Volume Spike**  
- **Price Breakout** (percentage change)  
- **RSI/MACD/VWAP** technical filters  
- **Earnings & News Date Range** filters to skip high-risk periods  
- **Stop Loss / Profit Target / Holding Period**  
- **Benchmark Comparison**  
- **Plotly**-based visualizations (candlestick chart + breakout markers, trade returns histogram)

The app is built in **Streamlit** for an interactive UI where users can tune parameters and see immediate results. All data is fetched via **Yahoo Finance** using the `yfinance` library.

---

## File Structure

advanced_breakout_project/ 
├── advanced_breakout_with_date_range_filter.py 
├── README.md 
├── docs/ 
│ └── Documentation.md 
├── requirements.txt 


- **`advanced_breakout_with_date_range_filter.py`**: The main Streamlit application that contains the entire advanced breakout logic, technical indicator computations, filters, and trade simulation.  
- **`README.md`**: High-level overview of setup, usage, and time spent.  
- **`docs/Documentation.md`**: (This file or similar) Detailed documentation of each component and how it works.

---

## Main Code Flow

1. **User Launches** the Streamlit app (e.g., `streamlit run advanced_breakout_with_date_range_filter.py`).  
2. **Sidebar**: The user inputs:
   - Ticker, date range, volume threshold, consecutive days, advanced filters, etc.  
   - Clicks **“Run Strategy.”**
3. **Data Download**: The app retrieves historical market data from Yahoo Finance.  
4. **Indicator Computations**: RSI, MACD, and VWAP calculations (if selected).  
5. **Breakout Detection**: 
   - Check volume vs. 20-day average.  
   - Check price breakout threshold.  
   - If `consecutive_days` > 1, require multiple bars in a row of volume > threshold.  
6. **Filters**:  
   - RSI min/max  
   - MACD bullish crossover  
   - VWAP (Close > VWAP)  
   - Earnings ± N days  
   - User-defined **date ranges** (skip trades within these ranges).  
7. **Trade Simulation**:  
   - For each breakout day, either buy at the same day’s close or next day’s open.  
   - Track exit if it hits stop loss or profit target, else hold until the end of `holding_period`.  
8. **Results**: 
   - Table of trades (buy/sell times, prices, % returns).  
   - Stats (average return, median return, total trades, win rate).  
   - Distribution histogram of trade returns.  
   - Candlestick chart with breakout markers.  
   - Benchmark performance vs. a user-chosen ticker (e.g., `^GSPC`).  
   - Download button for trade CSV.  

---

## Key Modules & Functions

### **Technical Indicator Calculations**

1. **`compute_rsi(df, period=14)`**  
   - Computes a **14-day RSI** using rolling average gains/losses.  
   - Adds a column `df['RSI']`.  

2. **`compute_macd(df, fastperiod=12, slowperiod=26, signalperiod=9)`**  
   - Calculates two EMAs (`EMA_fast`, `EMA_slow`) and derives `MACD = EMA_fast - EMA_slow`.  
   - Computes the `MACD_signal` line via an EMA of `MACD`.  
   - `MACD_hist = MACD - MACD_signal`.  

3. **`compute_daily_vwap(df)`**  
   - For daily data, approximates a 5-day rolling VWAP using typical price `(High+Low+Close)/3`.  
   - Adds `df['VWAP']`.  

---

### **Breakout Logic**

- **`mark_breakouts(df, vol_threshold, price_threshold, rolling_window, consecutive_days)`**  
  1. **Volume**: If day’s volume > `(vol_threshold% * 20-day average)`.  
  2. **Price**: If day’s close is up by `price_threshold%` from previous close.  
  3. **Consecutive Volume Days**: If `consecutive_days=2`, require 2 consecutive bars of volume above threshold, etc.  
  4. Sets `df['IsBreakout'] = True` only if both volume and price conditions (and consecutive day logic) are satisfied.

---

### **Filters**

1. **`apply_rsi_filter(df, rsi_min, rsi_max)`**  
   - Only keep breakouts where `rsi_min <= RSI <= rsi_max`.  

2. **`apply_macd_filter(df, use_macd)`**  
   - If `use_macd=True`, keep breakouts only if `MACD > MACD_signal`.  

3. **`apply_vwap_filter(df, use_vwap)`**  
   - If `use_vwap=True`, keep breakouts only if `Close > VWAP`.  

4. **`apply_earnings_filter(df, ticker, buffer_days)`**  
   - Attempts to fetch earnings dates via `yf.Ticker(ticker).get_earnings_dates(...)`.  
   - Excludes breakouts within ±`buffer_days` of those earnings dates.  

5. **News / Event Range Filter**  
   - **`parse_date_ranges(text_input)`**: Parse lines like “2023-07-01 to 2023-07-05” into `(start, end)` date tuples.  
   - **`apply_news_event_range_filter(df, date_ranges)`**: For each `(start, end)`, exclude breakouts on days that fall within that range.

---

### **Trade Simulation**

- **`simulate_trades(df, holding_period, stop_loss_pct, profit_target_pct, buy_next_day_open)`**  
  1. Iterate over each row `i` where `df['IsBreakout'] = True`.  
  2. **Entry**: 
     - If `buy_next_day_open=True`, buy at `i+1` open (if it exists).  
     - Otherwise, buy at `i` close.  
  3. Move forward up to `holding_period` bars.  
  4. If price hits `stop_loss_pct` or `profit_target_pct`, exit immediately.  
  5. If not triggered, exit on the final bar in that period.  
  6. Record buy/sell times, prices, and `%Return`.  

---

### **Benchmark**

- **`benchmark_performance(benchmark_ticker, start_date, end_date, interval)`**  
  - Downloads data for `benchmark_ticker`.  
  - Returns a simple buy-hold return from the first bar’s close to the last bar’s close.  

---

## Streamlit App Layout

1. **Sidebar**:  
   - Ticker, date range, volume threshold, consecutive days, advanced filters, etc.  
   - **“Run Strategy”** button triggers data download and calculations.  

2. **Main Page**:  
   - **Spinner** messages during data fetch and calculations.  
   - **Trades Table**: Buy/Sell times, returns.  
   - **Summary Stats**: average return, median return, total trades, win rate.  
   - **Distribution Plot**: histogram of trade returns.  
   - **Candlestick Chart**: price with breakout markers.  
   - **Benchmark Performance**: simple % return over the same period.  
   - **Download** button for CSV of trades.

---

## Data Handling & Caveats

1. **Yahoo Finance Limitations**  
   - Intraday data typically limited to ~60 days.  
   - Earnings data might only reflect upcoming events or partial historical coverage.  

2. **Date Matching**  
   - For intraday intervals, the code still uses `df.index.date` to compare with user-supplied date filters. This means partial day events can’t be precisely excluded unless further refined.  

3. **Combining Filters**  
   - When multiple filters are used (RSI, MACD, VWAP, multi-day volume, earnings, event dates, etc.), `df['IsBreakout']` is updated in a stepwise manner.  
   - Resulting breakouts can be drastically fewer if too many filters are active simultaneously.  

4. **Holding Period**  
   - We simply move forward `N` bars (rows). This includes weekends/holidays for daily data. If the user wants to skip those, more advanced logic is required.

---

## Extending the Code

1. **Position Sizing**: Use capital-based or risk-based position sizes instead of a simple “1 share.”  
2. **Trailing Stops**: Monitor the peak price since entry and exit if price falls X% from that peak.  
3. **Partial Take Profits**: Sell half the position at +5%, let the rest ride until +10% or trailing stop.  
4. **Machine Learning**: Incorporate a classifier to rank or filter out breakouts with lower probability of success.  
5. **Equity Curve**: Build a day-by-day portfolio equity curve and compare to a benchmark’s equity curve.  
6. **Transaction Costs**: Deduct commissions, slippage, or spread to simulate real-world results.
