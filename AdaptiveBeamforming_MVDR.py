import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Array and signal parameters
# -----------------------------
c = 3e8                  # Speed of light (m/s)
f = 10e9                 # Operating frequency (10 GHz)
lam = c / f              # Wavelength
d = lam / 2              # Element spacing
N = 8                    # Number of antenna elements

theta_signal = 20        # Desired signal (degrees)
theta_interf = -40       # Interference direction (degrees)
SNR_dB = 30              # Signal-to-noise ratio (dB)
INR_dB = 40              # Interference-to-noise ratio (dB)
snapshots = 1000         # Number of samples

# -----------------------------
# Steering vector function
# -----------------------------
def steering_vector(theta_deg):
    theta = np.deg2rad(theta_deg)
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * d / lam * n * np.sin(theta))

# ----------------------------- 
# Simulate received data
# -----------------------------
a_sig = steering_vector(theta_signal)
a_int = steering_vector(theta_interf)

signal = np.sqrt(10**(SNR_dB/10)) * np.exp(1j * 2 * np.pi * np.random.rand(snapshots))
interf = np.sqrt(10**(INR_dB/10)) * np.exp(1j * 2 * np.pi * np.random.rand(snapshots))
noise = (np.random.randn(N, snapshots) + 1j * np.random.randn(N, snapshots)) / np.sqrt(2)

# Total received data (each column = snapshot)
X = np.outer(a_sig, signal) + np.outer(a_int, interf) + noise

# -----------------------------
# MVDR (Capon) beamforming weights
# -----------------------------
R = (X @ X.conj().T) / snapshots   # Sample covariance matrix
a_desired = steering_vector(theta_signal)
w_mvdr = np.linalg.solve(R, a_desired)
w_mvdr = w_mvdr / (a_desired.conj().T @ np.linalg.solve(R, a_desired))

# -----------------------------
# Compute and plot spatial response
# -----------------------------
angles = np.linspace(-90, 90, 721)
AF = np.zeros_like(angles, dtype=float)

for i, th in enumerate(angles):
    a = steering_vector(th)
    AF[i] = np.abs(w_mvdr.conj().T @ a)

AF_dB = 20 * np.log10(AF / np.max(AF))

plt.figure(figsize=(8,5))
plt.plot(angles, AF_dB, linewidth=2)
plt.title(f'MVDR Adaptive Beamforming (Main Beam @ {theta_signal}°, Null @ {theta_interf}°)')
plt.xlabel('Angle (degrees)')
plt.ylabel('Normalized Gain (dB)')
plt.grid(True)
plt.ylim(-60, 0)
plt.show()
