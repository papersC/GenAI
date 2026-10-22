"""Background (cross) traffic in the medical network slice of each directed ISL.
Utilisation u_e(t) = geographic hotspot term (moves with the satellites)
               + AR(1) term + ramped congestion bursts (+ per-slot noise, added in the simulator)."""
import numpy as np

HOTSPOTS = [  # lat, lon, weight  (Gulf disaster surge, Europe, South Asia, Nile, East Africa)
    (25.0, 55.0, 1.0), (48.0, 10.0, 0.8), (22.0, 78.0, 0.9), (30.0, 31.0, 0.6), (0.0, 37.0, 0.5)]
SIGMA_KM = 1200.0


def hot_index(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    h = np.zeros_like(la)
    for hl, hlo, w in HOTSPOTS:
        a, b = np.radians(hl), np.radians(hlo)
        cosd = np.sin(la) * np.sin(a) + np.cos(la) * np.cos(a) * np.cos(lo - b)
        d = 6371.0 * np.arccos(np.clip(cosd, -1, 1))
        h += w * np.exp(-0.5 * (d / SIGMA_KM) ** 2)
    return np.clip(h, 0, 1)


def background(p, con, rng):
    """Return U[k, e] = mean background utilisation of directed ISL e during second k
    (k = 0..sim_s), excluding the per-slot noise."""
    Ts = p.sim_s + 2
    E = con.E
    U = np.zeros((Ts, E))
    # geographic term evaluated at transmitting satellite's sub-satellite point
    for k in range(Ts):
        pos = con.sat_pos(float(k))
        lat, lon = con.subsat(pos, float(k))
        h = hot_index(lat, lon)[con.src]
        U[k] = p.load_scale * (0.5 + 0.5 * h)
    # AR(1)
    phi = np.exp(-1.0 / p.ar_tau_s)
    x = rng.normal(0, p.ar_sigma, E)
    for k in range(Ts):
        U[k] += x
        x = phi * x + rng.normal(0, p.ar_sigma * np.sqrt(1 - phi ** 2), E)
    # ramped bursts (Poisson arrivals, rate scaled by hotspot index)
    h0 = np.clip((U[0] / p.load_scale - 0.5) / 0.5, 0, 1)
    rate = p.burst_rate_per_s * (0.5 + h0)
    nb = rng.poisson(rate * Ts)
    tt = np.arange(Ts, dtype=float)
    bursts = []
    for e in np.nonzero(nb)[0]:
        for _ in range(nb[e]):
            t0 = rng.uniform(-10, Ts)
            mag = rng.uniform(*p.burst_mag)
            hold = rng.exponential(p.burst_hold_mean_s)
            r = p.burst_ramp_s
            prof = np.clip((tt - t0) / r, 0, 1) * np.clip((t0 + r + hold + r - tt) / r, 0, 1)
            U[:, e] += mag * prof
            bursts.append((int(e), t0, mag, hold))
    return np.clip(U, 0.0, 1.3), bursts
