"""Walker-delta LEO constellation with +Grid ISLs, ground attachment and
logical Space-Data-Centre (SDC) clustering."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

R_E = 6371.0
MU = 398600.4418
C_KM_S = 299792.458
W_E = 7.2921159e-5


class Constellation:
    def __init__(self, p):
        self.p = p
        P, S = p.planes, p.sats_per_plane
        self.P, self.S, self.T = P, S, P * S
        self.r = R_E + p.altitude_km
        self.n = np.sqrt(MU / self.r ** 3)
        self.inc = np.radians(p.inclination_deg)
        pl = np.repeat(np.arange(P), S)
        sl = np.tile(np.arange(S), P)
        self.plane, self.slot = pl, sl
        self.raan = 2 * np.pi * pl / P
        self.u0 = 2 * np.pi * sl / S + 2 * np.pi * p.phasing * pl / self.T
        # +Grid ISLs (directed both ways)
        und = []
        for a in range(self.T):
            pa, sa = pl[a], sl[a]
            und.append((a, pa * S + (sa + 1) % S))          # intra-plane
            und.append((a, ((pa + 1) % P) * S + sa))        # inter-plane
        und = np.array(und)
        self.src = np.concatenate([und[:, 0], und[:, 1]])
        self.dst = np.concatenate([und[:, 1], und[:, 0]])
        self.E = len(self.src)
        self.edge_index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.src, self.dst))}
        # SDC clusters: contiguous blocks of planes x in-plane slots
        ca, cb = p.clusters_a, p.clusters_b
        self.cluster = (pl * ca // P) * cb + (sl * cb // S)
        self.M = ca * cb
        nb_cl = self.cluster[self.dst]
        border = np.zeros(self.T, bool)
        border[self.src[self.cluster[self.src] != nb_cl]] = True
        self.border = border  # GPU-equipped border satellites

    def sat_pos(self, t):
        u = self.u0 + self.n * t
        cO, sO, cu, su, ci, si = np.cos(self.raan), np.sin(self.raan), np.cos(u), np.sin(u), np.cos(self.inc), np.sin(self.inc)
        x = self.r * (cO * cu - sO * su * ci)
        y = self.r * (sO * cu + cO * su * ci)
        z = self.r * (su * si)
        return np.stack([x, y, z], 1)

    @staticmethod
    def ground_pos(lat, lon, t):
        lat, lon = np.radians(lat), np.radians(lon) + W_E * t
        return np.stack([R_E * np.cos(lat) * np.cos(lon), R_E * np.cos(lat) * np.sin(lon), R_E * np.sin(lat)], -1)

    def subsat(self, pos, t):
        lat = np.degrees(np.arcsin(pos[:, 2] / np.linalg.norm(pos, axis=1)))
        lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]) - W_E * t)
        lon = (lon + 180) % 360 - 180
        return lat, lon

    def isl_delay_ms(self, pos):
        d = np.linalg.norm(pos[self.src] - pos[self.dst], axis=1)
        return d / C_KM_S * 1e3

    def attach(self, lat, lon, pos, t):
        """Return (sat index, slant delay ms, elevation deg) for each ground point."""
        g = self.ground_pos(np.asarray(lat), np.asarray(lon), t)  # (G,3)
        v = pos[None, :, :] - g[:, None, :]
        rng = np.linalg.norm(v, axis=2)
        gh = g / np.linalg.norm(g, axis=1, keepdims=True)
        sin_el = np.einsum('gsk,gk->gs', v, gh) / rng
        el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
        best = np.argmax(el, axis=1)
        idx = np.arange(len(best))
        return best, rng[idx, best] / C_KM_S * 1e3, el[idx, best]

    def shortest_paths(self, delay_ms, sources):
        w = delay_ms + 0.01  # tiny hop penalty for tie breaking
        g = csr_matrix((w, (self.src, self.dst)), shape=(self.T, self.T))
        dist, pred = dijkstra(g, directed=True, indices=sources, return_predecessors=True)
        return dist, pred

    def path_edges(self, pred_row, src, dst):
        """Directed edge indices from src to dst using a predecessor row."""
        nodes = [dst]
        v = dst
        while v != src:
            v = pred_row[v]
            if v < 0:
                raise RuntimeError("unreachable")
            nodes.append(v)
        nodes = nodes[::-1]
        return [self.edge_index[(int(a), int(b))] for a, b in zip(nodes[:-1], nodes[1:])], nodes
