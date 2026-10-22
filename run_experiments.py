"""Run all experiment configurations x seeds in parallel; raw per-run JSON goes to results/raw/."""
import json, os, sys, time, traceback
from dataclasses import replace, asdict
from multiprocessing import Pool
from params import Params
import sim

SEEDS = list(range(1, 11))
SEM = ("SEM-S", "SEM-R", "SEM-PG", "SEM-PM", "SEM-OM")
BASE = Params()
CONFIGS = {"default": {}}
for n in (4, 8, 32, 48):
    CONFIGS[f"N{n}"] = dict(n_receivers=n)
for a, b in ((2, 2), (6, 6), (8, 8)):
    CONFIGS[f"M{a*b}"] = dict(clusters_a=a, clusters_b=b)
for L in (0.5, 0.6, 0.8, 0.9):
    CONFIGS[f"L{L}"] = dict(load_scale=L)
for r in (2.0, 6.0, 8.0):
    CONFIGS[f"rho{r:g}"] = dict(sem_gain=r, schemes=SEM)
for d in (0.5, 1.5):
    CONFIGS[f"dec{d:g}"] = dict(sem_dec_scale=d, schemes=SEM)
for gp in (0, 1, 6):
    CONFIGS[f"gpu{gp}"] = dict(gpu_per_cluster=gp, schemes=SEM)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw")


def job(arg):
    cfg, seed = arg
    fn = os.path.join(OUT, f"{cfg}_seed{seed}.json")
    if os.path.exists(fn):
        return fn, 0.0, "cached"
    p = replace(BASE, **CONFIGS[cfg])
    t = time.time()
    try:
        o = sim.run(p, seed, record_ts=(cfg == "default"))
    except Exception:
        return fn, 0.0, traceback.format_exc()
    o["config"] = cfg; o["params"] = asdict(p); o["wall_s"] = time.time() - t
    with open(fn, "w") as f:
        json.dump(o, f)
    return fn, o["wall_s"], "ok"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sel = sys.argv[1:] or list(CONFIGS)
    jobs = [(c, s) for c in sel for s in SEEDS]
    nproc = int(os.environ.get("NPROC", "6"))
    with Pool(nproc) as pool:
        for fn, dt, st in pool.imap_unordered(job, jobs):
            print(f"{os.path.basename(fn)} {dt:.1f}s {st[:300]}", flush=True)
