# Advanced Breakout Strategy with Filters

This project implements an **advanced breakout strategy** using **Streamlit** and **yfinance**, featuring:

1. **Multiple-Day Volume Spike**  
2. **Price Breakout Threshold**  
3. **RSI / MACD / VWAP** Filters  
4. **Earnings Date Filter**  
5. **News / Event Date Range Filter**  
6. **Stop Loss / Profit Target**  
7. **Holding Period** Logic  
8. **Benchmark Comparison**  

The result is a user-friendly web application that allows you to configure trading parameters, apply advanced filters, and visualize trades and performance. It can be easily run locally or deployed online on Streamlit Cloud (or similar services).

---

## Table of Contents

1. [Overview & Features](#overview--features)  
2. [Data Sources](#data-sources)  
3. [Setup & Installation](#setup--installation)  
4. [Usage](#usage)  
5. [Roadblocks & Challenges](#roadblocks--challenges)  
6. [Time Spent](#time-spent)  
7. [Future Improvements](#future-improvements)

---

## Overview & Features

This application showcases a multi-step approach to detect potential **breakout** trades in a given stock (or ETF/index). It includes:

- **Volume Spike**: Requires the day’s volume to be a certain percentage above the 20-day average. You can optionally require multiple consecutive days of above-threshold volume.  
- **Price Breakout**: Checks if today’s close is up by at least *X%* over the previous day’s close.  
- **RSI, MACD, VWAP Filters**: Optional technical filters to confirm momentum/trend.  
- **Earnings & News Filters**: Avoid trades around a company’s earnings release days or on user-defined date ranges (e.g., Fed announcements).  
- **Stop Loss & Profit Target**: Optionally exit early if the trade moves below/above these thresholds.  
- **Holding Period**: If no exit condition is triggered, exit after *N* bars (days or intraday bars).  
- **Benchmark**: Compare the strategy’s performance to a buy-and-hold of a chosen index (e.g., `^GSPC`).  

All logic is wrapped in a **Streamlit** interface with a sidebar for user inputs and a main area for:

- A **trades table**  
- **Summary statistics** (average return, median return, number of trades, win rate)  
- **Distribution plot** (histogram of trade returns)  
- **Price chart** with breakout markers  
- **Benchmark** results  

You can **download** the trades to a CSV file for further analysis.

---

## Data Sources

- **Yahoo Finance** (via `yfinance`):
  - Provides daily and intraday (15m, 30m, 1h, etc.) historical market data.  
  - We also fetch **earnings dates** using `Ticker.get_earnings_dates()` (coverage can be partial or upcoming only).

**Limitations**:

- **Intraday Data**: Typically limited to ~60 days of history.  
- **Historical Earnings**: Coverage is inconsistent; sometimes only future dates are available.

---

## Setup & Installation

1. **Clone or Download** this repository.  
2. (Optional) **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate      # on macOS/Linux
   # or
   venv\Scripts\activate         # on Windows

# Usage Instructions

### Enter Parameters in the Left Sidebar:
- **Ticker**: (e.g., `AAPL`)
- **Date Range**: (e.g., `start=2020-01-01`, `end=today`)
- **Interval**: (`daily` or `intraday`)
- **Volume Breakout Threshold (%)**
- **Price Breakout Threshold (%)**
- **Consecutive Volume Days**: (e.g., `2` means two consecutive days above the volume threshold)
- **Stop Loss**, **Profit Target**, **Holding Period**
- **RSI / MACD / VWAP Filters**
- **Earnings Filter**: Skip ±N days around earnings dates
- **News / Event Date Ranges**: Specify ranges to skip (e.g., `2023-07-01 to 2023-07-05`)
- **Benchmark Ticker**

### Steps:
1. Click **“Run Strategy.”**
2. Wait for:
   - Data to download
   - Strategy to simulate trades

### Results:
- **Trades Table**: 
  - `BuyTime`, `BuyPrice`, `SellTime`, `SellPrice`, `%Return`
- **Summary Metrics**:
  - Average return, median return, total trades, win rate
- **Distribution Plot**: Trade returns
- **Price Chart**: Breakout markers
- **Benchmark Performance**
- Option to **Download Trades CSV**

---

# Roadblocks & Challenges

### 1. **Earnings Data Coverage**
   - Yahoo Finance often provides partial or upcoming earnings data. Some tickers may lack data entirely.
   - The system handles this gracefully but cannot fully filter all historical earnings gaps without a more reliable dataset.

### 2. **Date/Time Alignment**
   - Intraday data has weekends and market holidays, requiring careful handling of missing days.

### 3. **Multiple Filters**
   - Applying RSI, MACD, VWAP, multi-day volume spikes, and news/earnings date exclusions can reduce signals to zero.

### 4. **Parsing Date Ranges**
   - Users can input lines like `2023-07-01 to 2023-07-05`, which the system parses into start and end dates.

### 5. **Intraday Limitations**
   - Historical intraday data from Yahoo rarely exceeds 60 days.

---

# Time Spent
- **Base Breakout Logic & Setup**: ~2 hours
- **Indicator & Filter Implementation**: ~3 hours
- **UI Enhancements & Testing**: ~2 hours
- **Documentation & Cleanup**: ~1 hour
- **Total**: ~8 hours

---

# Future Improvements

### 1. **Better Intraday VWAP**
   - Fetch higher-resolution data (e.g., 1-minute bars) for accurate intraday VWAP calculations.

### 2. **Historical Earnings Data**
   - Use a paid data source for more robust earnings coverage.

### 3. **Stop/Profit & Trailing**
   - Add trailing stops or partial take-profit features.

### 4. **Machine Learning**
   - Train a model to filter breakouts with a higher likelihood of success.

### 5. **Equity Curve Tracking**
   - Track day-by-day portfolio equity for deeper performance metrics (e.g., Sharpe ratio, max drawdown).
