import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def markov_transition_field(time_series, n_bins=5):
    # Step 1: Discretize the time series into `n_bins` states
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretized_series = discretizer.fit_transform(time_series.reshape(-1, 1)).flatten()

    # Step 2: Calculate state transition probabilities
    n_states = n_bins
    transition_matrix = np.zeros((n_states, n_states))

    for i in range(len(discretized_series) - 1):
        current_state = int(discretized_series[i])
        next_state = int(discretized_series[i + 1])
        transition_matrix[current_state, next_state] += 1

    # Normalize to obtain probabilities
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)

    # Step 3: Construct the MTF
    mtf = np.zeros((len(discretized_series), len(discretized_series)))

    for i in range(len(discretized_series)):
        for j in range(len(discretized_series)):
            if i <= j:
                current_state = int(discretized_series[i])
                next_state = int(discretized_series[j])
                mtf[i, j] = transition_matrix[current_state, next_state]

    return mtf


# Example time series data
time_series_data = np.sin(np.linspace(0, 2 * np.pi, 100))

# Generate MTF
mtf = markov_transition_field(time_series_data, n_bins=5)

# Visualize the MTF
plt.imshow(mtf, cmap='hot', interpolation='nearest')
plt.title("Markov Transition Field")
plt.colorbar()
plt.show()
