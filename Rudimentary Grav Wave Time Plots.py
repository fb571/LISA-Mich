import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv  # Bessel function of the first kind
from scipy.signal import get_window

ω0 = 0.5*0.03795*(2*np.pi) #Initial Frequency
e0 = 0.3 #Initial Eccentricity
m1 = 0.32 #Mass of Primary
m2 = 0.37 #Mass of Secondary
r = 2.7846e09 #Distance of Binary from Observer
I = 0 #Angle of Inclination
φ = 0 #Angle of Pericentre

G = 0.0003965 #Gravitational Constant
c = 20.04 #Speed of Light
a0 = (G*(m1+m2)/ω0**2)**(1/3) #Initial Semimajor Axis

#Number of timesteps and the time interval value
N = 4000
h = 0.1

#"Independent" variables (at least independent to begin with)
# Units of 0.01AU = 100s = 1 solar mass = 1

y0 = np.array([a0,e0])

def f(y):
    a = y[0]
    e = y[1]
    coeff = (G**3) * m1 * m2 * (m1 + m2) / (c**5)
    aprime = (-64/5 * coeff *(1 + (73/24)*e**2 + (37/96)*e**4)/(a**3 * (1 - e**2)**(7/2)))
    eprime = (-304/15 * coeff * e *(1 + (121/304)*e**2) /(a**4 * (1 - e**2)**(5/2)))
    derivative = np.array([aprime,eprime])
    return derivative

def rk4(y0,func,N,h):
    M = np.zeros([N+1,2])
    M[0,:] = y0
    for i in range(1,N+1):
        y = M[i-1,:]
        k1 = h*func(y)
        k2 = h*func(y+0.5*k1)
        k3 = h*func(y+0.5*k2)
        k4 = h*func(y+k3)
        M[i,:] = y + (k1 + 2*k2 + 2*k3 + k4)/6
    return M

Y = rk4(y0,f,N,h)

a = Y[:,0]
e = Y[:,1]
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

# Now compute the summed harmonic contributions (vectorized)
# sum over harmonics (axis=1) to collapse to shape (N+1,)
prefactor = - (mu * (omega**2) * G) / (r * c**4)   # shape (N+1,)

hplus_vec  = prefactor * np.sum(Hp_coef * cos_phase, axis=1)
hcross_vec = prefactor * np.sum(Hc_coef * sin_phase, axis=1)

# hplus_vec and hcross_vec are the full time series (length N+1)
hpl = hplus_vec.copy()
hcr = hcross_vec.copy()
noise = np.random.normal(0, 0, N+1)

plt.figure(figsize=(8,5))
plt.plot(t, hcr+noise, label='Hx (cross)', color='blue')
plt.plot(t, hpl+noise, label='Hp (plus)', color='red')
plt.legend()
plt.xlabel(r"Time (Seconds $\times 10^2$)")
plt.ylabel("Strain amplitude")
plt.title("Gravitational Wave Polarizations Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()