# Auto-converted from PatternRecognition.ipynb
# This file was generated to be runnable in an offline environment with mocks.

# ----------------------------------------
#@title Instal and Import Libraries
# [SHELL SKIPPED] !pip install yfinance pandas scipy numpy plotly -q
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------
#@title STEP 1 — Fetch & Clean Stock Data
# ==========================================
def get_clean_financial_data(ticker, start_date, end_date):
    """Fetch and clean financial data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Standardize column names
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            data.columns = [col.capitalize() if col.lower() in [e.lower() for e in expected_cols] else col for col in data.columns]
            if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                print(f"Warning: Columns might not be as expected. Found: {data.columns.tolist()}")

        # Handle missing values
        data = data.ffill().bfill()

        # Remove timezone info
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        print(f"✅ Successfully fetched {len(data)} days of data for {ticker}")
        return data

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# ----------------------------------------
#@title STEP 2 — Detect Peaks & Troughs
# ==========================================
def find_price_peaks_troughs(prices, height=None, distance=5, prominence=None):
    """
    Find peaks and troughs in price data using scipy.signal
    """
    # Find peaks (local maxima)
    peaks_indices, _ = find_peaks(
        prices, height=height, distance=distance, prominence=prominence
    )

    # Find troughs (local minima)
    troughs_indices, _ = find_peaks(
        -prices, height=height, distance=distance, prominence=prominence
    )

    return peaks_indices, troughs_indices


def detect_trend_reversals(data, distance=10, prominence_ratio=0.02):
    """
    Detect trend reversals by finding significant peaks and troughs
    """
    close_prices = data['Close'].values
    avg_price = np.mean(close_prices)
    prominence = avg_price * prominence_ratio

    peaks, troughs = find_price_peaks_troughs(
        close_prices, distance=distance, prominence=prominence
    )
    return peaks, troughs

# ----------------------------------------
#@title STEP 3 — Detect Chart Patterns
# ==========================================
def detect_chart_patterns(data, peaks, troughs, tolerance=0.03):
    """
    Detect common chart patterns (Double Top, Double Bottom, Triple Top, Triple Bottom)
    """
    patterns = []
    prices = data['Close'].values

    def is_close(a, b):
        return abs(a - b) / ((a + b) / 2) <= tolerance

    # --- Double Top ---
    for i in range(len(peaks) - 1):
        first_top = prices[peaks[i]]
        second_top = prices[peaks[i + 1]]
        troughs_between = [t for t in troughs if peaks[i] < t < peaks[i + 1]]
        if not troughs_between:
            continue
        trough_between = prices[min(troughs_between, key=lambda x: prices[x])]
        if is_close(first_top, second_top) and trough_between < min(first_top, second_top) * (1 - tolerance):
            patterns.append({
                "pattern": "Double Top",
                "first_top": data.index[peaks[i]],
                "second_top": data.index[peaks[i + 1]],
                "trough": data.index[min(troughs_between, key=lambda x: prices[x])]
            })

    # --- Double Bottom ---
    for i in range(len(troughs) - 1):
        first_bottom = prices[troughs[i]]
        second_bottom = prices[troughs[i + 1]]
        peaks_between = [p for p in peaks if troughs[i] < p < troughs[i + 1]]
        if not peaks_between:
            continue
        peak_between = prices[max(peaks_between, key=lambda x: prices[x])]
        if is_close(first_bottom, second_bottom) and peak_between > max(first_bottom, second_bottom) * (1 + tolerance):
            patterns.append({
                "pattern": "Double Bottom",
                "first_bottom": data.index[troughs[i]],
                "second_bottom": data.index[troughs[i + 1]],
                "peak": data.index[max(peaks_between, key=lambda x: prices[x])]
            })

    # --- Triple Top ---
    for i in range(len(peaks) - 2):
        top1, top2, top3 = prices[peaks[i]], prices[peaks[i + 1]], prices[peaks[i + 2]]
        if is_close(top1, top2) and is_close(top2, top3):
            patterns.append({
                "pattern": "Triple Top",
                "tops": [data.index[peaks[i]], data.index[peaks[i + 1]], data.index[peaks[i + 2]]]
            })

    # --- Triple Bottom ---
    for i in range(len(troughs) - 2):
        b1, b2, b3 = prices[troughs[i]], prices[troughs[i + 1]], prices[troughs[i + 2]]
        if is_close(b1, b2) and is_close(b2, b3):
            patterns.append({
                "pattern": "Triple Bottom",
                "bottoms": [data.index[troughs[i]], data.index[troughs[i + 1]], data.index[troughs[i + 2]]]
            })

    return patterns

# ----------------------------------------
#@title STEP 4 — Visualization with Patterns
# ==========================================
def plot_patterns(data, peaks, troughs, patterns, ticker):
    """
    Plot price chart with peaks/troughs and highlight detected patterns.
    """
    fig = go.Figure()

    # Add close price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='gray', width=1.2),
        opacity=0.8
    ))

    # Mark peaks
    fig.add_trace(go.Scatter(
        x=data.index[peaks],
        y=data['Close'].iloc[peaks],
        mode='markers',
        name='Peaks',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ))

    # Mark troughs
    fig.add_trace(go.Scatter(
        x=data.index[troughs],
        y=data['Close'].iloc[troughs],
        mode='markers',
        name='Troughs',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ))

    # Track which patterns we've already added to legend
    patterns_in_legend = set()

    # Highlight detected patterns
    for p in patterns:
        pattern = p["pattern"]

        # --- Double Top ---
        if pattern == "Double Top":
            x_vals = [p["first_top"], p["second_top"]]
            y_vals = [data.loc[x_vals[0], 'Close'], data.loc[x_vals[1], 'Close']]
            show_legend = pattern not in patterns_in_legend
            if show_legend:
                patterns_in_legend.add(pattern)

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=10, symbol='circle'),
                name='Double Top',
                showlegend=show_legend,
                legendgroup='Double Top'
            ))

        # --- Double Bottom ---
        elif pattern == "Double Bottom":
            x_vals = [p["first_bottom"], p["second_bottom"]]
            y_vals = [data.loc[x_vals[0], 'Close'], data.loc[x_vals[1], 'Close']]
            show_legend = pattern not in patterns_in_legend
            if show_legend:
                patterns_in_legend.add(pattern)

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=10, symbol='circle'),
                name='Double Bottom',
                showlegend=show_legend,
                legendgroup='Double Bottom'
            ))

        # --- Triple Top ---
        elif pattern == "Triple Top":
            x_vals = p["tops"]
            y_vals = [data.loc[x, 'Close'] for x in x_vals]
            show_legend = pattern not in patterns_in_legend
            if show_legend:
                patterns_in_legend.add(pattern)

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                line=dict(color='purple', width=3),
                marker=dict(size=10, symbol='circle'),
                name='Triple Top',
                showlegend=show_legend,
                legendgroup='Triple Top'
            ))

        # --- Triple Bottom ---
        elif pattern == "Triple Bottom":
            x_vals = p["bottoms"]
            y_vals = [data.loc[x, 'Close'] for x in x_vals]
            show_legend = pattern not in patterns_in_legend
            if show_legend:
                patterns_in_legend.add(pattern)

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=10, symbol='circle'),
                name='Triple Bottom',
                showlegend=show_legend,
                legendgroup='Triple Bottom'
            ))

    fig.update_layout(
        title=f"{ticker} – Chart Patterns with Peaks and Troughs",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        height=600
    )

    fig.show()

# ----------------------------------------
#@title STEP 5 — Main Analysis Function
# ==========================================
def analyze_stock_peaks_troughs(ticker, start_date, end_date, distance=10):
    data = get_clean_financial_data(ticker, start_date, end_date)
    if data.empty:
        print("No data available for analysis")
        return

    # Detect turning points
    peaks, troughs = detect_trend_reversals(data, distance=distance)

    print(f"📊 Analysis Results for {ticker}:")
    print(f"   Total data points: {len(data)}")
    print(f"   Peaks detected: {len(peaks)}")
    print(f"   Troughs detected: {len(troughs)}")

    # Detect chart patterns
    patterns = detect_chart_patterns(data, peaks, troughs)

    # Remove the detailed pattern printing - just show count summary
    if patterns:
        pattern_counts = {}
        for p in patterns:
            pattern_name = p['pattern']
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

        print("\n📈 Detected Chart Patterns Summary:")
        for pattern_name, count in pattern_counts.items():
            print(f"   {pattern_name}: {count} instances")
    else:
        print("\nNo classic chart patterns detected.")

    # Visualize everything
    plot_patterns(data, peaks, troughs, patterns, ticker)

    return data, peaks, troughs, patterns


# ==========================================
# STEP 6 — Run Example
# ==========================================
if __name__ == "__main__":
    ticker = "NVDA"
    start_date = "2023-01-01"
    end_date = "2025-10-15"

    data, peaks, troughs, patterns = analyze_stock_peaks_troughs(ticker, start_date, end_date)

