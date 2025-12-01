import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv  # Bessel function of the first kind
from scipy.signal import get_window

ω0 = 0.00311*(2*np.pi) #Initial Frequency
e0 = 0.000 #Initial Eccentricity
m1 = 0.55*2e30 #Mass of Primary
m2 = 0.27*2e30 #Mass of Secondary
r = 1.54e20 #Distance of Binary from Observer
I = 0.66 #Angle of Inclination
φ = 0 #Angle of Pericentre

G = 6.67e-11 #Gravitational Constant
c = 3e8 #Speed of Light
a0 = (G*(m1+m2)/ω0**2)**(1/3) #Initial Semimajor Axis

#Number of timesteps and the time interval value
N = 4000000
h = 32
T = N*h

#"Independent" variables (at least independent to begin with)
# Units of 1m = 1s = 1kg = 1

a = np.full(N+1,a0)
e = np.full(N+1,e0)
t = np.arange(N+1)*h

nmax = 10
n = np.arange(1, nmax+1)

n_grid = n[None, :]                 # shape (1, nmax) -> will broadcast to (N+1,nmax)

# Broadcast grids
# ne[i,n] = n * e[i]
ne = e[:, None] * n[None, :]        # shape (N+1, nmax)

# Precompute Bessel terms for orders n-2, n-1, n+1, n+2
# Build order grid matching ne shape
n_grid = n[None, :]                 # shape (1, nmax) -> will broadcast to (N+1,nmax)

J_m2 = jv(n_grid - 2, ne)  # jv(n-2, n*e_i) -> shape (N+1,nmax)
J_m1 = jv(n_grid - 1, ne)
J_p1 = jv(n_grid + 1, ne)
J_p2 = jv(n_grid + 2, ne)

# Compute A_n, B_n, C_n for all times at once
# a[:,None] broadcasts a[i] across harmonics
a2_over_n = (a[:, None]**2) / n[None, :]   # shape (N+1, nmax)

An = a2_over_n * (J_m2 - J_p2 - 2.0 * e[:, None] * (J_m1 - J_p1))
Bn = a2_over_n * (1.0 - e[:, None]**2) * (J_p2 - J_m2)
Cn = a2_over_n * np.sqrt(1.0 - e[:, None]**2) * (J_p2 + J_m2 - e[:, None] * (J_p1 + J_m1))

# Compute omega for each time (note: omega depends on a[i])
omega = np.sqrt(G * (m1 + m2) / a**3)      # shape (N+1,)

# Compute the per-time-per-harmonic phase: n * omega_i * t_i
# For each time i and harmonic n: phase[i,n] = n * omega[i] * t[i]
phase = (omega * t)[:, None] * n[None, :]  # shape (N+1, nmax)

cos_phase = np.cos(phase)
sin_phase = np.sin(phase)

# Precompute geometric angle combinations (scalars)
Cphi = np.cos(φ); Sphi = np.sin(φ)
C2phi = np.cos(2.0*φ); S2phi = np.sin(2.0*φ)
CI = np.cos(I); SI = np.sin(I)

# Coefficient arrays multiplying the trigonometric time dependence
Hp_coef = (
    An * (Cphi**2 - Sphi**2 * CI**2)
    + Bn * (Sphi**2 - Cphi**2 * CI**2)
    - Cn * (np.sin(2.0*φ)) * (1.0 + CI**2)
)   # shape (N+1, nmax)

Hc_coef = (
    2.0 * C2phi * CI * Cn
    + (An - Bn) * S2phi * SI
)   # shape (N+1, nmax)

# Mass factor
mu = (m1 * m2) / (m1 + m2)
mc = ((m1*m2)**(3/5))/((m1+m2)**(1/5))

# Now compute the summed harmonic contributions (vectorized)
# sum over harmonics (axis=1) to collapse to shape (N+1,)
prefactor = - (mu * (omega**2) * G) / (r * c**4)   # shape (N+1,)
prefactor1 = - 2*((G*mc)**(5/3))/(r * c**4 * a**2)   # shape (N+1,)

hplus_vec  = prefactor * np.sum(Hp_coef * cos_phase, axis=1)
hcross_vec = prefactor * np.sum(Hc_coef * sin_phase, axis=1)

# hplus_vec and hcross_vec are the full time series (length N+1)
hpl = hplus_vec.copy()
hcr = hcross_vec.copy()

w = get_window('hann', N+1)
W = np.sum(w**2)/(N+1)

# Compute FFTs (normalized)
X = h*np.fft.rfft((hcr)*w) / np.sqrt(W)
Y = h*np.fft.rfft((hpl)*w) / np.sqrt(W)

#LISA Noise Curve
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

# Frequency axis (properly scaled)
λ = np.fft.rfftfreq(N+1, d=h)

#The noise added to INDIVIDUAL REAL/IMAGINARY PARTS (hence we divide by 4 rather than 2)
f = λ[1:]
LISAnoisePSD = Snwithc(f)

characstrainX = 4*abs(X)[1:] * np.sqrt(2*f/T)
characstrainY = 4*abs(Y)[1:] * np.sqrt(2*f/T)
fnoisecurve = np.sqrt(f*LISAnoisePSD)

plt.figure(figsize=(8,5))
plt.plot(np.log10(f), np.log10(characstrainY), label='Characteristic Strain in Hcross', color='blue')
plt.plot(np.log10(f), np.log10(characstrainX), label='Characteristic Strain in Hplus', color='red')
plt.plot(np.log10(f), np.log10(fnoisecurve), label='Characteristic Strain in Noise', color='green')
plt.legend()
plt.xlabel("log10(Frequency (Hz))")
plt.ylabel("log10(Characteristic Strain)")
plt.title("Characteristic Strain for HM Cancri and Noise")
plt.grid(True)                     # major grid
plt.minorticks_on()               # enable minor ticks
plt.grid(which='minor', linestyle=':', alpha=0.4)   # minor grid
plt.xlim(-3.7, -2)
plt.ylim(-21, -18)
plt.tight_layout()
plt.show()
