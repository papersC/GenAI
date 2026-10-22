"""Time-slotted flow-level simulator of clinical video multicast over a LEO ISL mesh.
One call of run(params, seed) simulates every scheme on identical geometry,
subscriptions and background traffic (common random numbers)."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from constellation import Constellation
from traffic import background
import policies as P


class Geo:
    """Per-epoch geometry: attachments, shortest-path trees, path incidences."""
    pass


def build_geo(p, con, t, src_ll, rx_ll, subs, rx_kind):
    g = Geo()
    pos = con.sat_pos(t)
    d_isl = con.isl_delay_ms(pos)
    s_sat, s_up, s_el = con.attach(src_ll[:, 0], src_ll[:, 1], pos, t)
    r_sat, r_dn, r_el = con.attach(rx_ll[:, 0], rx_ll[:, 1], pos, t)
    dist, pred = con.shortest_paths(d_isl, s_sat)
    g.nrx = len(rx_ll)
    pairs = [(s, i) for s in range(len(src_ll)) for i in range(g.nrx) if subs[s, i]]
    g.npairs = len(pairs)
    g.pair_src = np.array([s for s, _ in pairs]); g.pair_rx = np.array([i for _, i in pairs])
    g.pairs_of_rx = [[q for q, (s, i) in enumerate(pairs) if i == r] for r in range(g.nrx)]
    g.rx_dl = np.where(rx_kind == 0, p.icu_downlink_mbps, p.mob_downlink_mbps)
    g.rx_gpu = np.where(rx_kind == 0, p.icu_gpu_units, p.mob_gpu_units)
    g.pair_w = np.where(rx_kind[g.pair_rx] == 0, p.w_icu, 0.0)
    g.paths, g.prop = [], np.zeros(g.npairs)
    g.trees = []
    for s in range(len(src_ll)):
        root = int(s_sat[s])
        tr = dict(root=root, children={root: []}, rx_at={root: []}, parent={}, pedge={}, order=[root])
        g.trees.append(tr)
    for q, (s, i) in enumerate(pairs):
        tr = g.trees[s]
        edges, nodes = con.path_edges(pred[s], int(s_sat[s]), int(r_sat[i]))
        g.paths.append(np.array(edges, int))
        g.prop[q] = s_up[s] + r_dn[i] + d_isl[edges].sum()
        for a_, b_, e in zip(nodes[:-1], nodes[1:], edges):
            if b_ not in tr['parent']:
                tr['parent'][b_] = a_; tr['pedge'][b_] = e
                tr['children'].setdefault(a_, []).append(b_)
                tr['children'].setdefault(b_, []); tr['rx_at'].setdefault(b_, [])
                tr['order'].append(b_)
        tr['rx_at'].setdefault(nodes[-1], []).append(q)
    # order is BFS-compatible because nodes are appended in path order (parents first)
    rows = np.concatenate([[q] * len(e) for q, e in enumerate(g.paths)]) if g.npairs else np.array([])
    cols = np.concatenate(g.paths) if g.npairs else np.array([])
    g.A = csr_matrix((np.ones(len(rows)), (rows.astype(int), cols.astype(int))), shape=(g.npairs, con.E))
    g.hops = np.array([len(e) for e in g.paths])
    g.min_elev = float(min(s_el.min(), r_el.min()))
    return g


def run(p, seed, record_ts=False):
    rng = np.random.default_rng(seed)
    con = Constellation(p)
    # scenario: sources, receivers, subscriptions
    sb, rb = p.src_box, p.rx_box
    src_ll = np.c_[rng.uniform(sb[0], sb[1], p.n_sources), rng.uniform(sb[2], sb[3], p.n_sources)]
    rx_ll = np.c_[rng.uniform(rb[0], rb[1], p.n_receivers), rng.uniform(rb[2], rb[3], p.n_receivers)]
    n_icu = int(round(p.frac_icu * p.n_receivers))
    rx_kind = np.r_[np.zeros(n_icu, int), np.ones(p.n_receivers - n_icu, int)]
    subs = rng.random((p.n_sources, p.n_receivers)) < p.p_subscribe
    for i in range(p.n_receivers):
        if not subs[:, i].any():
            subs[rng.integers(p.n_sources), i] = True
    U, bursts = background(p, con, rng)
    noise_seed = int(rng.integers(1 << 31))
    C = p.isl_capacity_mbps
    nE = p.sim_s
    spe = int(round(p.epoch_s / p.slot_s))
    geos = [build_geo(p, con, float(k), src_ll, rx_ll, subs, rx_kind) for k in range(nE)]
    npairs = geos[0].npairs
    Bmax = p.buffer_ms * 1e-3 * C  # Mbit
    ser_ms = p.pkt_bytes * 8 / (C * 1e6) * 1e3
    out = {'seed': seed, 'npairs': npairs, 'n_bursts': len(bursts),
           'mean_hops': float(np.mean([g.hops.mean() for g in geos])),
           'min_elev': float(min(g.min_elev for g in geos)),
           'mean_prop_ms': float(np.mean([g.prop.mean() for g in geos])), 'schemes': {}}
    for name in p.schemes:
        sem = name.startswith("SEM")
        coding = P.Coding(p, sem)
        mt = P.max_tiers(p, geos[0], coding)
        react = P.Reactive(p, mt, coding) if name.endswith("-R") else None
        nrng = np.random.default_rng(noise_seed)
        q = np.zeros(con.E)   # backlog (Mbit)
        lvl = None; trd = np.zeros(con.E)
        meas = None
        rec = dict(Lb=[], Lf=[], loss=[], tier=[], Q=[], isl=[], gpu=[], solve=[], status=[])
        ts = dict(isl=[], Q=[], Lb_mean=[], Lb_max=[], ontime=[])
        for k in range(nE):
            g = geos[k]
            # ---------- predictions ----------
            if lvl is None:
                ubg_hat = U[k]  # first epoch: assume known initial state
            else:
                ubg_hat = lvl + trd
            ubg_or = 0.5 * (U[k] + U[k + 1])
            Lhat = p.fixed_ms + g.prop + g.hops * ser_ms + g.A @ (q / C * 1e3)
            # ---------- decision ----------
            st, dt = "", 0.0
            if name == "UC-H265":
                tiers = mt.copy(); er = P.unicast(p, con, g, coding, tiers); nd = np.zeros(npairs, int); gpu = 0
                dl = tiers
            else:
                if name.endswith("-S"):
                    des = mt
                elif name.endswith("-R"):
                    des = react.step(meas)
                elif name.endswith("-PG"):
                    t0 = time.perf_counter(); des = P.greedy(p, con, g, coding, ubg_hat, Lhat, mt); dt = time.perf_counter() - t0
                elif name.endswith("-PM"):
                    des, dt, st = P.milp(p, con, g, coding, ubg_hat, Lhat, mt)
                elif name.endswith("-OM"):
                    des, dt, st = P.milp(p, con, g, coding, ubg_or, Lhat, mt)
                er, dl, nd, gpu = P.project(p, con, g, coding, des)
            # receiver decoder contention (only binds if a policy exceeds the budget)
            load = np.zeros(g.nrx)
            np.add.at(load, g.pair_rx, coding.units[dl])
            fac = np.where(g.rx_gpu[g.pair_rx] > 0, np.maximum(1.0, load[g.pair_rx] / np.maximum(g.rx_gpu[g.pair_rx], 1)), 1.0)
            extra = coding.dec_ms[dl] * fac + p.requant_ms * nd
            # ---------- slots ----------
            Qd = np.zeros((spe, con.E)); Lp = np.zeros((spe, con.E)); bgm = np.zeros(con.E); um = np.zeros(con.E)
            for j in range(spe):
                f = j / spe
                ub = (1 - f) * U[k] + f * U[k + 1] + nrng.normal(0, p.fast_sigma, con.E)
                ub = np.clip(ub, 0, None)
                A = (ub * C + er) * p.slot_s     # Mbit arriving in slot
                qn = q + A - C * p.slot_s
                drop = np.clip(qn - Bmax, 0, None)
                qn = np.clip(qn, 0, Bmax)
                Lp[j] = np.where(A > 0, drop / np.maximum(A, 1e-12), 0)
                Qd[j] = 0.5 * (q + qn) / C * 1e3
                q = qn
                bgm += ub; um += ub + er / C
            bgm /= spe; um /= spe
            base = (p.fixed_ms + g.prop + g.hops * ser_ms)[:, None]
            Lb = base + (g.A @ Qd.T)                                         # pairs x slots
            loss = 1 - np.exp(g.A @ np.log(np.clip(1 - Lp.T, 1e-9, 1)))
            Lf = Lb + extra[:, None]
            # ---------- measurement for next epoch ----------
            if lvl is None:
                lvl = bgm.copy()
            else:
                nl = p.holt_alpha * bgm + (1 - p.holt_alpha) * (lvl + trd)
                trd = p.holt_beta * (nl - lvl) + (1 - p.holt_beta) * trd
                lvl = nl
            umax = np.array([um[e].max() if len(e) else 0.0 for e in g.paths])
            meas = dict(loss=loss.mean(1), Lfull_max=Lf.max(1), umax=umax)
            # ---------- metrics ----------
            ontime_b = Lb <= p.deadline_ms
            ontime_f = Lf <= p.deadline_ms
            Qs = np.where((dl[:, None] > 0) & ontime_f, coding.Q[dl][:, None],
                          np.where(ontime_b, coding.Q[0], 0.0)) * (1 - loss)
            isl = er.sum()
            if k >= p.warmup_s:
                rec['Lb'].append(Lb); rec['Lf'].append(np.where(dl[:, None] > 0, Lf, np.nan)); rec['loss'].append(loss)
                rec['tier'].append(np.repeat(dl[:, None], spe, 1)); rec['Q'].append(Qs)
                rec['isl'].append(isl); rec['gpu'].append(gpu)
                if dt: rec['solve'].append(dt)
                if st: rec['status'].append(st)
            if record_ts:
                ts['isl'].append(isl); ts['Q'].append(float(Qs.mean())); ts['Lb_mean'].append(float(Lb.mean()))
                ts['Lb_max'].append(float(Lb.max())); ts['ontime'].append(float(((1 - loss) * ontime_b).mean()))
        Lb = np.concatenate(rec['Lb'], 1); loss = np.concatenate(rec['loss'], 1)
        Lf = np.concatenate(rec['Lf'], 1); Qs = np.concatenate(rec['Q'], 1); tier = np.concatenate(rec['tier'], 1)
        icu = (rx_kind[geos[0].pair_rx] == 0)
        ontime_b = (Lb <= p.deadline_ms)
        res = dict(
            isl_mbps=float(np.mean(rec['isl'])),
            lat_base_mean=float(Lb.mean()), lat_base_p95=float(np.percentile(Lb, 95)), lat_base_p99=float(np.percentile(Lb, 99)),
            lat_full_mean=float(np.nanmean(Lf)) if np.isfinite(Lf).any() else float('nan'),
            lat_full_p95=float(np.nanpercentile(Lf, 95)) if np.isfinite(Lf).any() else float('nan'),
            base_deadline_ratio=float(ontime_b.mean()),
            struct_infeasible=float(np.mean(np.concatenate([np.repeat(((p.fixed_ms + g.prop + g.hops * ser_ms) > p.deadline_ms)[:, None], spe, 1) for g in geos[p.warmup_s:]], 1))),
            delivery_ratio=float((1 - loss).mean()),
            ontime_delivery=float(((1 - loss) * ontime_b).mean()),
            enh_ontime=float(np.nanmean(np.where(tier > 0, (Lf <= p.deadline_ms) * (1 - loss), np.nan))) if (tier > 0).any() else float('nan'),
            quality=float(Qs.mean()), quality_icu=float(Qs[icu].mean()) if icu.any() else float('nan'),
            mean_tier_icu=float(tier[icu].mean()) if icu.any() else 0.0,
            jitter_ms=float(np.mean(np.abs(np.diff(Lb, axis=1)))),
            gpu_mean=float(np.mean(rec['gpu'])),
            solve_mean_s=float(np.mean(rec['solve'])) if rec['solve'] else 0.0,
            solve_max_s=float(np.max(rec['solve'])) if rec['solve'] else 0.0,
            n_nonoptimal=int(sum(1 for s in rec['status'] if s != 'Optimal')),
        )
        if record_ts:
            res['ts'] = ts
        out['schemes'][name] = res
    return out
