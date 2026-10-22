"""Adaptation policies: projection onto multicast-tree constraints, static,
reactive (threshold + hysteresis), predictive greedy and predictive MILP (PuLP/HiGHS)."""
import time
import numpy as np
import pulp


class Coding:
    """Rate/quality/latency ladder for conventional (SHVC-style) or semantic coding."""
    def __init__(self, p, semantic):
        self.sem = semantic
        conv = np.array(p.conv_cum_mbps)
        qf = 100 * (1 - np.exp(-conv / p.q_r0_mbps))
        self.K = len(conv) - 1
        if semantic:
            self.cum = p.base_mbps + (conv - p.base_mbps) / p.sem_gain
            self.Q = qf.copy(); self.Q[1:] = np.minimum(qf[1:], p.sem_qcap)
            self.dec_ms = p.sem_dec_scale * np.array(p.sem_dec_ms) + np.r_[0, [p.sem_enc_extra_ms] * self.K]
            self.units = np.array(p.sem_dec_units)
        else:
            self.cum = conv
            self.Q = qf
            self.dec_ms = np.r_[0.0, [p.conv_enh_dec_ms] * self.K]
            self.units = np.zeros(self.K + 1, int)


def max_tiers(p, geo, coding):
    """Per-pair maximum tier allowed by receiver downlink and decoder budgets (static split)."""
    mt = np.zeros(geo.npairs, int)
    for i in range(geo.nrx):
        prs = geo.pairs_of_rx[i]
        if not prs:
            continue
        dl, gu = geo.rx_dl[i], geo.rx_gpu[i]
        best = 0
        for k in range(coding.K, -1, -1):
            if len(prs) * coding.cum[k] <= dl + 1e-9 and (not coding.sem or len(prs) * coding.units[k] <= gu):
                best = k; break
        mt[prs] = best
    return mt


def project(p, con, geo, coding, desired):
    """Map desired per-pair tiers to tree-feasible edge tiers.
    Conventional layers can be dropped anywhere for free; semantic tokens can be
    dropped entirely anywhere, but re-quantised to a lower non-zero tier only at
    GPU-equipped border satellites with spare SDC GPU budget."""
    gpu_left = np.full(con.M, p.gpu_per_cluster, float)
    edge_rate = np.zeros(con.E)
    delivered = np.zeros(geo.npairs, int)
    ndrops = np.zeros(geo.npairs, int)
    for s, tr in enumerate(geo.trees):
        # subtree demand
        D = {v: 0 for v in tr['order']}
        for v in reversed(tr['order']):
            d = max([desired[q] for q in tr['rx_at'][v]] + [D[c] for c in tr['children'][v]] + [0])
            D[v] = d
        tier = {tr['root']: D[tr['root']]}
        drops_above = {tr['root']: 0}
        for v in tr['order']:
            tv = tier[v]
            gpu_ok = con.border[v]
            cl = con.cluster[v]
            outs = [(c, D[c], None) for c in tr['children'][v]] + [(None, desired[q], q) for q in tr['rx_at'][v]]
            for c, dc, q in outs:
                dr = 0
                if dc == 0 or tv == 0:
                    tc = 0
                elif not coding.sem:
                    tc = dc                      # SHVC layers: free drop anywhere
                elif dc >= tv:
                    tc = tv
                elif gpu_ok and gpu_left[cl] >= 1:
                    gpu_left[cl] -= 1; tc, dr = dc, 1   # onboard token re-quantisation
                else:
                    tc = tv                      # cannot shed here: forward as is
                if c is not None:
                    tier[c] = tc; drops_above[c] = drops_above[v] + dr
                    edge_rate[tr['pedge'][c]] += coding.cum[tc]
                else:
                    delivered[q] = tc; ndrops[q] = drops_above[v] + dr
    return edge_rate, delivered, ndrops, p.gpu_per_cluster * con.M - gpu_left.sum()


def unicast(p, con, geo, coding, tiers):
    edge_rate = np.zeros(con.E)
    for q in range(geo.npairs):
        edge_rate[geo.paths[q]] += coding.cum[tiers[q]]
    return edge_rate


class Reactive:
    """Per-pair threshold policy with hysteresis (non-predictive)."""
    def __init__(self, p, mt, coding):
        self.p, self.mt, self.dec = p, mt, coding.dec_ms
        self.t = mt.copy()
        self.good = np.zeros(len(mt), int)

    def step(self, meas):
        p = self.p
        if meas is None:
            return self.t.copy()
        bad = (meas['loss'] > 0.01) | (meas['Lfull_max'] > p.deadline_ms) | (meas['umax'] > p.react_hi_util)
        nxt = np.minimum(self.t + 1, len(self.dec) - 1)
        head = meas['Lfull_max'] + (self.dec[nxt] - self.dec[self.t]) + 10.0   # 10 ms safety margin
        ok = (meas['loss'] <= 1e-3) & (head <= p.deadline_ms) & (meas['umax'] < p.react_lo_util)
        self.good = np.where(ok, self.good + 1, 0)
        self.t = np.where(bad, np.maximum(self.t - 1, 0), self.t)
        up = (self.good >= p.react_up_epochs) & (self.t < self.mt)
        self.t = np.where(up, self.t + 1, self.t)
        self.good[up] = 0
        return self.t.copy()


def _feasible(p, con, geo, coding, desired, ubg, Lhat):
    er, dl, nd, _ = project(p, con, geo, coding, desired)
    tot = er + ubg * p.isl_capacity_mbps
    return er, dl, nd, tot


def greedy(p, con, geo, coding, ubg, Lhat, mt):
    """Predictive greedy: round-robin one-tier upgrades (highest weight first) that keep
    every touched ISL below eta*C (forecast), the latency bound and receiver budgets."""
    des = np.zeros(geo.npairs, int)
    er0, dl0, nd0, _ = project(p, con, geo, coding, des)
    cap = p.eta * p.isl_capacity_mbps
    order = np.argsort(-geo.pair_w, kind='stable')
    changed = True
    while changed:
        changed = False
        for q in order:
            if des[q] >= mt[q]:
                continue
            trial = des.copy(); trial[q] += 1
            er, dl, nd, tot = _feasible(p, con, geo, coding, trial, ubg, Lhat)
            inc = er > er0 + 1e-9
            if np.any(tot[inc] > cap):
                continue
            bad = False
            for qq in np.nonzero((dl != dl0) | (nd != nd0))[0]:
                i = geo.pair_rx[qq]; prs = geo.pairs_of_rx[i]
                if coding.sem and coding.units[dl[prs]].sum() > geo.rx_gpu[i]:
                    bad = True; break
                if coding.cum[dl[prs]].sum() > geo.rx_dl[i] + 1e-9:
                    bad = True; break
                if dl[qq] > 0 and Lhat[qq] + coding.dec_ms[dl[qq]] + p.requant_ms * nd[qq] > p.deadline_ms:
                    bad = True; break
            if bad:
                continue
            des, er0, dl0, nd0 = trial, er, dl, nd
            changed = True
    return des


def milp(p, con, geo, coding, ubg, Lhat, mt):
    """Predictive joint tier/shedding MILP (Sec. IV). Returns per-pair desired tiers,
    solve time and status."""
    K = coding.K
    dr = np.diff(coding.cum)
    dQ = np.diff(coding.Q)
    ddec = np.diff(coding.dec_ms)
    dun = np.diff(coding.units)
    C = p.isl_capacity_mbps
    prob = pulp.LpProblem("tier", pulp.LpMaximize)
    obj = []
    edge_terms = {}   # ISL e -> list of affine terms (Mbps)
    gpu_terms = {j: [] for j in range(con.M)}
    a = {}
    lat_w = {q: [] for q in range(geo.npairs)}
    for s, tr in enumerate(geo.trees):
        z = {}
        # edge keys: ('U',) uplink into root; ('N', child) ISL into child; ('R', q) leaf
        def mk(key):
            z[key] = [None] + [pulp.LpVariable(f"z_{s}_{key[0]}_{key[1]}_{k}", cat="Binary") for k in range(1, K + 1)]
            for k in range(2, K + 1):
                prob.addConstraint(z[key][k] <= z[key][k - 1])
            return z[key]
        mk(('U', 0))
        incoming = {tr['root']: ('U', 0)}
        wvars = {}
        for v in tr['order']:
            pk = incoming[v]
            outs = [('N', c) for c in tr['children'][v]] + [('R', q) for q in tr['rx_at'][v]]
            for key in outs:
                zc = mk(key)
                for k in range(1, K + 1):
                    prob += zc[k] <= z[pk][k]
                if key[0] == 'N':
                    incoming[key[1]] = key
                    e = tr['pedge'][key[1]]
                    edge_terms.setdefault(e, []).append(float(coding.cum[0]))
                    edge_terms[e].append(pulp.lpSum(dr[k - 1] * zc[k] for k in range(1, K + 1)))
                    obj.append(-p.lambda_bw * pulp.lpSum(dr[k - 1] * zc[k] for k in range(1, K + 1)))
                if coding.sem and K >= 2:
                    if con.border[v]:
                        w = pulp.LpVariable(f"w_{s}_{key[0]}_{key[1]}", 0, 1)
                        for k in range(2, K + 1):
                            prob += w >= z[pk][k] - zc[k] + zc[1] - 1
                        gpu_terms[con.cluster[v]].append(w)
                        wvars[key] = w
                        obj.append(-0.01 * w)
                    else:
                        for k in range(2, K + 1):
                            prob += zc[k] >= z[pk][k] - (1 - zc[1])
        for v in tr['order']:
            for q in tr['rx_at'][v]:
                a[q] = z[('R', q)]
                # GPU re-quantisations on the path of q
                node = v; ws = []
                if ('R', q) in wvars:
                    ws.append(wvars[('R', q)])
                while node != tr['root']:
                    key = ('N', node)
                    if key in wvars:
                        ws.append(wvars[key])
                    node = tr['parent'][node]
                lat_w[q] = ws
    sig = []
    for e, terms in edge_terms.items():
        sg = pulp.LpVariable(f"sig_{e}", 0)
        sig.append(sg)
        prob += pulp.lpSum(terms) + float(ubg[e]) * C - sg <= p.eta * C
    for j, ws in gpu_terms.items():
        if ws:
            prob += pulp.lpSum(ws) <= p.gpu_per_cluster
    for i in range(geo.nrx):
        prs = geo.pairs_of_rx[i]
        if not prs:
            continue
        prob += pulp.lpSum(dr[k - 1] * a[q][k] for q in prs for k in range(1, K + 1)) <= max(0.0, geo.rx_dl[i] - len(prs) * coding.cum[0])
        if coding.sem:
            prob += pulp.lpSum(dun[k - 1] * a[q][k] for q in prs for k in range(1, K + 1)) <= geo.rx_gpu[i]
    for q in range(geo.npairs):
        for k in range(mt[q] + 1, K + 1):
            prob += a[q][k] == 0
        slack = p.deadline_ms - Lhat[q]
        prob += (pulp.lpSum(ddec[k - 1] * a[q][k] for k in range(1, K + 1))
                 + p.requant_ms * pulp.lpSum(lat_w[q]) <= slack + 1000 * (1 - a[q][1]))
        if slack < coding.dec_ms[1]:
            prob += a[q][1] == 0
        obj.append(geo.pair_w[q] * pulp.lpSum(dQ[k - 1] * a[q][k] for k in range(1, K + 1)))
    obj.append(-1000 * pulp.lpSum(sig))
    prob += pulp.lpSum(obj)
    t0 = time.perf_counter()
    solver = pulp.HiGHS(msg=False, timeLimit=p.milp_time_limit_s, gapRel=p.milp_gap)
    prob.solve(solver)
    dt = time.perf_counter() - t0
    st = pulp.LpStatus[prob.status]
    des = np.zeros(geo.npairs, int)
    for q in range(geo.npairs):
        des[q] = int(round(sum((a[q][k].value() or 0) for k in range(1, K + 1))))
    return des, dt, st
