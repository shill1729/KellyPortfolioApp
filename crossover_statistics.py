import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_success_runs(data):
    success_runs_up = []
    success_runs_down = []

    run_length_up = 0
    run_length_down = 0

    for i in range(1, len(data)):
        if data['adj_close'].iloc[i] > data['adj_close'].iloc[i - 1]:
            if run_length_down > 0:
                success_runs_down.append(run_length_down)
                run_length_down = 0
            run_length_up += 1
        elif data['adj_close'].iloc[i] < data['adj_close'].iloc[i - 1]:
            if run_length_up > 0:
                success_runs_up.append(run_length_up)
                run_length_up = 0
            run_length_down += 1
        else:
            if run_length_up > 0:
                success_runs_up.append(run_length_up)
                run_length_up = 0
            if run_length_down > 0:
                success_runs_down.append(run_length_down)
                run_length_down = 0

    if run_length_up > 0:
        success_runs_up.append(run_length_up)
    if run_length_down > 0:
        success_runs_down.append(run_length_down)

    return success_runs_up, success_runs_down


def calculate_moving_averages(data, short_window, long_window):
    data['short_ema'] = data['adj_close'].ewm(span=short_window, adjust=False).mean()
    data['long_ema'] = data['adj_close'].ewm(span=long_window, adjust=False).mean()
    return data


def identify_crossovers(data):
    data['signal'] = 0.0
    data.loc[data.index[short_window:], 'signal'] = np.where(
        data['short_ema'].iloc[short_window:] > data['long_ema'].iloc[short_window:], 1.0, 0.0
    )
    data['positions'] = data['signal'].diff()
    return data


def evaluate_accuracy(data, trend_length=6):
    correct_predictions = 0
    total_crossovers = len(data[data['positions'] != 0])

    for i in range(1, len(data) - 6):  # Ensure we have enough data points for checking 5 days ahead
        if data['positions'].iloc[i] == 1:  # Bullish crossover
            if data['adj_close'].iloc[i + 1:i + trend_length].mean() > data['adj_close'].iloc[i]:  # Trend up over
                # next 5 days
                correct_predictions += 1
        elif data['positions'].iloc[i] == -1:  # Bearish crossover
            if data['adj_close'].iloc[i + 1:i + trend_length].mean() < data['adj_close'].iloc[i]:  # Trend down over
                # next 5 days
                correct_predictions += 1

    accuracy = correct_predictions / total_crossovers if total_crossovers else 0
    return correct_predictions, total_crossovers, accuracy


# Sample usage
if __name__ == "__main__":
    # Load your data here. For example, using a CSV file:
    data = pd.read_csv('~/downloads/spy.csv')

    # Ensure your data has an 'adj_close' column with adjusted closing prices

    short_window = 30
    long_window = 250
    trend_length = 21

    data = calculate_moving_averages(data, short_window, long_window)
    data = identify_crossovers(data)

    correct_predictions, total_crossovers, accuracy = evaluate_accuracy(data, trend_length)

    print(f"Correct predictions: {correct_predictions}")
    print(f"Total crossovers: {total_crossovers}")
    print(f"Accuracy: {accuracy:.2f}")

    # Analyzing succes runs
    success_runs_up, success_runs_down = analyze_success_runs(data)
    # Print summary statistics
    print(f"Success runs up: mean = {np.mean(success_runs_up):.2f}, median = {np.median(success_runs_up):.2f}, "
          f"max = {np.max(success_runs_up):.2f}")
    print(f"Success runs down: mean = {np.mean(success_runs_down):.2f}, median = {np.median(success_runs_down):.2f}, "
          f"max = {np.max(success_runs_down):.2f}")

    # Plotting the data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))

    ax1.plot(data['adj_close'], label='Adjusted Close Price')
    ax1.plot(data['short_ema'], label=f'{short_window}-day EMA')
    ax1.plot(data['long_ema'], label=f'{long_window}-day EMA')
    ax1.plot(data[data['positions'] == 1.0].index, data['short_ema'][data['positions'] == 1.0], '^', markersize=10,
             color='g', lw=0, label='Buy Signal')
    ax1.plot(data[data['positions'] == -1.0].index, data['short_ema'][data['positions'] == -1.0], 'v', markersize=10,
             color='r', lw=0, label='Sell Signal')
    ax1.set_title('EMA Crossover Strategy')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot histogram of success runs up
    ax2.hist(success_runs_up, bins=range(1, max(success_runs_up) + 2), edgecolor='black')
    ax2.set_title('Histogram of Success Runs Up')
    ax2.set_xlabel('Run Length (days)')
    ax2.set_ylabel('Frequency')

    # Plot histogram of success runs down
    ax3.hist(success_runs_down, bins=range(1, max(success_runs_down) + 2), edgecolor='black')
    ax3.set_title('Histogram of Success Runs Down')
    ax3.set_xlabel('Run Length (days)')
    ax3.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()