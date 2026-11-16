"""
Physical world model: reference trajectory, vessel traffic, and Kalman filter.
"""
import numpy as np
from .config import DT, SPEED, RECT, SIGMA_P, SIGMA_V, F, B, Q, R0


def make_reference(n_steps):
    """Generate a rectangular patrol reference trajectory."""
    wps = np.array([[0, 0], [RECT, 0], [RECT, RECT], [0, RECT], [0, 0]], float)
    legs = []
    for a, b in zip(wps[:-1], wps[1:]):
        k = max(1, int(round(np.linalg.norm(b - a) / (SPEED * DT))))
        legs += [a + (b - a) * (i / k) for i in range(k)]
    legs = np.array(legs)
    if len(legs) < n_steps:
        legs = np.vstack([legs, np.repeat(legs[-1:], n_steps - len(legs), axis=0)])
    return legs[:n_steps]


def route_distance(p):
    """Cross-track deviation: distance of each point in *p* (N x 2) from the
    planned patrol polyline (the rectangle)."""
    wps = np.array([[0, 0], [RECT, 0], [RECT, RECT], [0, RECT], [0, 0]], float)
    p = np.atleast_2d(p)
    best = np.full(len(p), np.inf)
    for a, b in zip(wps[:-1], wps[1:]):
        ab = b - a
        u = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
        d = np.linalg.norm(p - (a + u[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def vessel_tracks(n_steps, n_vessels, rng):
    """Generate straight-line vessel tracks."""
    vs = []
    for _ in range(n_vessels):
        p0 = rng.uniform(-200, RECT + 200, 2)
        hdg = rng.uniform(0, 2 * np.pi)
        spd = rng.uniform(2.0, 6.0)
        vel = spd * np.array([np.cos(hdg), np.sin(hdg)])
        vs.append(p0[None, :] + np.outer(np.arange(n_steps) * DT, vel))
    return np.array(vs)


class KF:
    """Constant-velocity Kalman filter with position+velocity state."""

    def __init__(self, x0):
        self.x = x0.astype(float).copy()
        self.P = np.diag([4.0, 4.0, 1.0, 1.0])

    def predict(self, a, Qm=None):
        """Propagate with input acceleration *a*; *Qm* overrides the process
        noise (used for the constant-velocity fallback when the IMU is isolated)."""
        self.x = F @ self.x + B @ a
        self.P = F @ self.P @ F.T + (Q if Qm is None else Qm)

    def innovation(self, z):
        nu = z - self.x
        S = self.P + R0
        return float(nu @ np.linalg.solve(S, nu))

    def update(self, z, Rscale=1.0):
        S = self.P + R0 * Rscale
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(4) - K) @ self.P

    def update_pos(self, z2, Rp):
        """Position-only update: maritime absolute-position fix (GNSS- and
        IMU-independent), used as the fallback when GNSS is isolated."""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = H @ self.P @ H.T + Rp * np.eye(2)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z2 - self.x[:2])
        self.P = (np.eye(4) - K @ H) @ self.P

    def pos_std(self):
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
