import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

# ============================================================
# 1. Time grid
# ============================================================

np.random.seed(123)

label = "freq_peak_with_noise"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

N = 1000
dt = 50
t = np.arange(N) * dt

# ============================================================
# True signal parameters
# ============================================================

true_f0 = 0.00611

ω0 = true_f0*(np.pi) #Initial Frequency
m1 = 0.5*2e30 #Mass of Primary
m2 = 0.5*2e30 #Mass of Secondary
r = 1.5e19 #Distance of Binary from Observer

G = 6.67e-11 #Gravitational Constant
c = 3e8 #Speed of Light
a0 = (G*(m1+m2)/ω0**2)**(1/3) #Initial Semimajor Axis

mu = (m1 * m2) / (m1 + m2)
prefactor = 4 * ω0**2 * a0**2 * (mu * G) / (r * c**4)

true_A = prefactor

true_phi = 1.2

# ============================================================
# 3. Time-domain signal model
# ============================================================

def gw_time_domain(t, A, f0, phi):
    return A * np.sin(2 * np.pi * f0 * t + phi)

h_true = gw_time_domain(t, true_A, true_f0, true_phi)

# ============================================================
# 4. Window + FFT (NO time-domain noise)
# ============================================================

window = get_window("hann", N)
W = np.sum(window**2) / N
norm = np.sqrt(W)

h_fft = dt * np.fft.rfft(h_true * window) / norm
freqs = np.fft.rfftfreq(N, d=dt)

print(true_A)

# ============================================================
# 5. Add noise directly in frequency domain
# ============================================================

L = 2.5e9
fstar = 0.01909
α = 0.138
β = -221
κ = 521
γ = 1680
fk = 0.00113

def Snwithc(x):
    f = x
    Poms = ((1.5e-11)**2)*(1+(0.002/f)**4)
    Pacc = ((3e-15)**2)*(1+(0.0004/f)**2)*(1+(f/0.008)**4)
    Scf = 9e-45 * (f ** (-7/3)) * np.exp(-f**α + β*f*np.sin(κ*f)) * (1 + np.tanh(γ * (fk - f)))
    return (10/(3*L**2))*(Poms + 4*Pacc/(2*np.pi*f)**4)*(1+(6/10)*(f/fstar)**2) + Scf

freqs = np.fft.rfftfreq(N, d=dt)
b = np.zeros(len(h_fft))
b[1:] = np.sqrt(Snwithc(freqs[1:])*N*dt/2)

true_noise = b   # frequency-domain noise scale

noise_fft = true_noise * (
    np.random.randn(len(h_fft)) +
    1j * np.random.randn(len(h_fft))
)

data_fft = h_fft + noise_fft
print(data_fft)
# ============================================================
# 6. Plot FFT (oscillatory sidelobes preserved)
# ============================================================

plt.figure(figsize=(8,5))
plt.plot(freqs, np.log10(np.abs(data_fft.real)), label="Re(data FFT)")
plt.plot(freqs, np.log10(np.abs(data_fft.imag)), label="Im(data FFT)")
plt.axvline(true_f0, color="k", ls="--", label="True f0")
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("FFT amplitude")
plt.tight_layout()
plt.show()

# ============================================================
# 7. Likelihood: complex frequency-domain Gaussian
# ============================================================

class FFTGWLikelihood(bilby.Likelihood):
    def __init__(self, t, data_fft, window, dt, freq_min=0.005, freq_max=0.007):
        super().__init__()
        self.t = t
        self.data_fft = data_fft
        self.window = window
        self.dt = dt
        self.norm = np.sqrt(np.sum(window**2) / len(window))
        
        # FFT frequencies
        self.freqs = np.fft.rfftfreq(len(t), d=dt)
        
        # Frequency mask to restrict likelihood evaluation
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_mask = (self.freqs >= freq_min) & (self.freqs <= freq_max)
        
        # Required by bilby
        self.parameters = dict(A=None, f0=None, phi=None)

    def log_likelihood(self):
        A = self.parameters["A"]
        f0 = self.parameters["f0"]
        phi = self.parameters["phi"]

        # Time-domain model
        h_td = gw_time_domain(self.t, A, f0, phi)

        # Window + FFT
        h_fft = self.dt * np.fft.rfft(h_td * self.window) / self.norm

        # Residual only in the frequency range of interest
        diff = self.data_fft[self.freq_mask] - h_fft[self.freq_mask]

        # Using constant noise scale as in your example
        noise = true_noise

        chi2 = (diff.real**2 + diff.imag**2) / noise[500:700]**2
        return -0.5 * np.sum(chi2)

# ============================================================
# 8. Priors
# ============================================================

priors = dict(
    A=bilby.core.prior.Uniform(0.0, 3e-21, "A"),
    f0=bilby.core.prior.Uniform(0.005, 0.007, "f0"),
    phi=bilby.core.prior.Uniform(0, 2*np.pi, "phi"),
)

# ============================================================
# 9. Run bilby
# ============================================================

likelihood = FFTGWLikelihood(t, data_fft, window, dt)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=800,
    print_progress=True,
    outdir="outdir",
    label="fft_gw_freq_noise",
)

result.plot_corner()