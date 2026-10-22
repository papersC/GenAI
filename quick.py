import sys, time, json
from params import Params
import sim
p = Params(sim_s=int(sys.argv[1]) if len(sys.argv)>1 else 30)
t=time.time(); o = sim.run(p, 1); print("time", time.time()-t)
print({k:v for k,v in o.items() if k!='schemes'})
for n,r in o['schemes'].items():
    print(f"{n:10s} isl={r['isl_mbps']:7.1f} Lb={r['lat_base_mean']:6.1f} p95={r['lat_base_p95']:6.1f} dl={r['base_deadline_ratio']:.3f} deliv={r['delivery_ratio']:.4f} ontime={r['ontime_delivery']:.3f} Q={r['quality']:5.1f} Qicu={r['quality_icu']:5.1f} tier={r['mean_tier_icu']:.2f} enh={r['enh_ontime']:.3f} gpu={r['gpu_mean']:.1f} inf={r['struct_infeasible']:.3f} solve={r['solve_mean_s']:.3f} nonopt={r['n_nonoptimal']}")
