import numpy as np
from params import Params
import sim, policies as P
from constellation import Constellation
p=Params()
rng=np.random.default_rng(1)
con=Constellation(p)
sb, rb = p.src_box, p.rx_box
src_ll = np.c_[rng.uniform(sb[0], sb[1], p.n_sources), rng.uniform(sb[2], sb[3], p.n_sources)]
rx_ll = np.c_[rng.uniform(rb[0], rb[1], p.n_receivers), rng.uniform(rb[2], rb[3], p.n_receivers)]
n_icu = int(round(p.frac_icu * p.n_receivers)); rx_kind = np.r_[np.zeros(n_icu, int), np.ones(p.n_receivers - n_icu, int)]
subs = rng.random((p.n_sources, p.n_receivers)) < p.p_subscribe
g=sim.build_geo(p,con,30.0,src_ll,rx_ll,subs,rx_kind)
cod=P.Coding(p,True); mt=P.max_tiers(p,g,cod)
ser=p.pkt_bytes*8/(p.isl_capacity_mbps*1e6)*1e3
Lhat=p.fixed_ms+g.prop+g.hops*ser
ub=np.full(con.E,0.5)
dg=P.greedy(p,con,g,cod,ub,Lhat,mt); dm,_,_=P.milp(p,con,g,cod,ub,Lhat,mt)
for name,d in (('G',dg),('M',dm)):
    er,dl,nd,gpu=P.project(p,con,g,cod,d)
    Lf=Lhat+cod.dec_ms[dl]+p.requant_ms*nd
    print(name,'des',d,'\n  dl',dl,'\n  nd',nd,'\n  late',np.round(np.where(dl>0,Lf-150,0),1),'gpu',gpu, 'Q',(cod.Q[dl]*g.pair_w).sum())
print('Lhat',np.round(Lhat,1)); print('mt',mt)
