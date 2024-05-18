import numpy as np
from scipy.integrate import odeint as scipy_odeint
import os

# Directory setup
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def hindmarsh_rose(state, t, a, b, c, d, r, s, x0, I):
    x, y, z = state
    dx = y - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - y
    dz = r * (s * (x - x0) - z)
    return [dx, dy, dz]


# Parameters
a = 1.0
b = 3.0
c = 1.0
d = 5.0
r = 0.5
s = 4.0  # Adjusted to a higher value for stability
x0 = -1.5  # Adjusted for stability
I = 3.0

init_state = [2, -8, 0.5]

T = 20
n_seq = 1000
seq_len = 1500 
t = np.linspace(0, T, seq_len)

# Generate sequences
sequences = []
for _ in range(n_seq):
    noise_intensity = 0.01  # Adjust noise intensity to your preference
    noise = np.random.normal(loc=0, scale=noise_intensity, size=(3,))
    init_state_with_noise = init_state + noise
    states = scipy_odeint(hindmarsh_rose, init_state_with_noise, t, args=(a, b, c, d, r, s, x0, I))
    sequences.append(states)

# Stack sequences
all_sequences = np.stack(sequences)

print(all_sequences.shape)

# Save the dataset
np.save(os.path.join(current_directory, 'hindmarsh_rose_dataset.npy'), all_sequences)
