"""Paired (across seeds) comparisons quoted in the paper text."""
from analyze import load, paired
runs = load()
Q = [("default","SEM-PM","MC-SVC-S","isl_mbps"),("default","SEM-PM","UC-H265","isl_mbps"),("default","SEM-PM","MC-SVC-PM","isl_mbps"),
     ("default","SEM-PM","MC-SVC-PM","quality_icu"),("default","SEM-PM","SEM-R","quality_icu"),("default","SEM-PM","SEM-PG","quality_icu"),
     ("default","SEM-OM","SEM-PM","quality_icu"),("default","SEM-PM","MC-SVC-S","ontime_delivery"),("default","SEM-PM","SEM-S","enh_ontime"),
     ("L0.8","SEM-PM","MC-SVC-PM","quality_icu"),("L0.8","SEM-PM","SEM-R","quality_icu"),("L0.8","SEM-PM","SEM-S","ontime_delivery"),("L0.8","SEM-PM","MC-SVC-S","ontime_delivery"),
     ("L0.9","SEM-PM","SEM-S","ontime_delivery"),("L0.9","SEM-PM","MC-SVC-S","ontime_delivery"),
     ("N48","SEM-PM","SEM-PG","quality_icu"),("N48","SEM-PM","UC-H265","isl_mbps"),("N48","SEM-PM","MC-SVC-S","isl_mbps"),
     ("rho2","SEM-PM","SEM-S","isl_mbps"),("gpu0","SEM-PM","SEM-PG","quality_icu")]
with open("results/aggregated/paired_extra.txt","w") as f:
    for c,a,b,m in Q:
        (d,hd,n),(rl,hr,_),pv = paired(runs,c,a,b,m)
        s=f"[{c}] {a} - {b} [{m}]: diff={d:.4f}±{hd:.4f} rel={rl:.2f}%±{hr:.2f}% p={pv:.2e} n={n}"
        print(s); f.write(s+"\n")
# cross-config: gpu0 vs default for SEM-PM (same seeds, same scenario) 
import numpy as np
from scipy import stats
d=[ [r for r in runs['default'] if r['seed']==s][0]['schemes']['SEM-PM']['quality_icu'] - [r for r in runs['gpu0'] if r['seed']==s][0]['schemes']['SEM-PM']['quality_icu'] for s in range(1,11)]
t=stats.ttest_1samp(d,0); h=stats.t.ppf(.975,9)*np.std(d,ddof=1)/np.sqrt(10)
s=f"[default vs gpu0] SEM-PM quality_icu G=3 minus G=0: diff={np.mean(d):.3f}±{h:.3f} p={t.pvalue:.2e}"
print(s); open("results/aggregated/paired_extra.txt","a").write(s+"\n")
