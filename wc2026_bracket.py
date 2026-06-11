"""
WC2026 simulator — REAL FIFA bracket (fixes the Elo-seed / sequential-pairing bug).
R32 slot template (M73-M88), third-place pools, and feed-forward tree (M89-M104)
are the official FIFA structure from the tournament regulations.
"""
import numpy as np, pandas as pd
from collections import defaultdict
RNG = np.random.default_rng(42)

def build_elo(df, base=1500, home_adv=65):
    R=defaultdict(lambda:base)
    def k_of(t):
        t=str(t)
        if 'World Cup' in t and 'qualification' not in t: return 60
        if 'qualification' in t or 'Nations League' in t: return 40
        if 'Friendly' in t: return 20
        return 30
    for r in df.dropna(subset=['home_score','away_score']).sort_values('date').itertuples():
        ha=0 if r.neutral else home_adv
        ea=1/(1+10**(-(R[r.home_team]-R[r.away_team]+ha)/400))
        gd=abs(r.home_score-r.away_score); mult=1 if gd<=1 else (1.5 if gd==2 else 1.75+(gd-3)/8)
        sa=1 if r.home_score>r.away_score else (0 if r.home_score<r.away_score else .5)
        K=k_of(r.tournament)*mult
        R[r.home_team]+=K*(sa-ea); R[r.away_team]-=K*(sa-ea)
    return dict(R)

GROUPS={'A':['Mexico','South Korea','Czech Republic','South Africa'],'B':['Switzerland','Canada','Bosnia and Herzegovina','Qatar'],'C':['United States','Paraguay','Australia','Turkey'],'D':['Brazil','Morocco','Scotland','Haiti'],'E':['Germany','Ecuador','Ivory Coast','Curaçao'],'F':['Netherlands','Japan','Sweden','Tunisia'],'G':['Belgium','Egypt','Iran','New Zealand'],'H':['Spain','Uruguay','Saudi Arabia','Cape Verde'],'I':['France','Senegal','Norway','Iraq'],'J':['Argentina','Austria','Algeria','Jordan'],'K':['Portugal','Colombia','Uzbekistan','DR Congo'],'L':['England','Croatia','Ghana','Panama']}
BASE,BETA=1.36,0.45

def sim_match(a,b,R):
    d=(R.get(a,1500)-R.get(b,1500))/400
    return RNG.poisson(BASE*np.exp(BETA*d)), RNG.poisson(BASE*np.exp(-BETA*d))
def ko(a,b,R):
    x,y=sim_match(a,b,R)
    if x>y: return a
    if y>x: return b
    return a if RNG.random()<1/(1+10**(-(R.get(a,1500)-R.get(b,1500))/400)) else b
def sim_group(teams,R):
    pts=defaultdict(int);gf=defaultdict(int);ga=defaultdict(int)
    for i in range(4):
        for j in range(i+1,4):
            a,b=teams[i],teams[j];x,y=sim_match(a,b,R)
            gf[a]+=x;ga[a]+=y;gf[b]+=y;ga[b]+=x
            if x>y:pts[a]+=3
            elif y>x:pts[b]+=3
            else:pts[a]+=1;pts[b]+=1
    rk=sorted(teams,key=lambda t:(pts[t],gf[t]-ga[t],gf[t],RNG.random()),reverse=True)
    return rk,{t:(pts[t],gf[t]-ga[t],gf[t]) for t in teams}

# ---- REAL FIFA R32 template ----
# winner-vs-third slots: match -> (winner group, allowed third-place group pool)
THIRD_SLOTS={'M74':('E',set('ABCDF')),'M77':('I',set('CDFGH')),'M79':('A',set('CEFHI')),
             'M80':('L',set('EHIJK')),'M81':('D',set('BEFIJ')),'M82':('G',set('AEHIJ')),
             'M85':('B',set('EFGIJ')),'M87':('K',set('DEIJL'))}
# the other 8 R32 matches (no third place)
FIXED_R32={'M73':('RU','A','RU','B'),'M75':('W','F','RU','C'),'M76':('W','C','RU','F'),
           'M78':('RU','E','RU','I'),'M83':('RU','K','RU','L'),'M84':('W','H','RU','J'),
           'M86':('W','J','RU','H'),'M88':('RU','D','RU','G')}
# feed-forward tree
R16=[('M89','M74','M77'),('M90','M73','M75'),('M91','M76','M78'),('M92','M79','M80'),
     ('M93','M83','M84'),('M94','M81','M82'),('M95','M86','M88'),('M96','M85','M87')]
QF=[('M97','M89','M90'),('M98','M93','M94'),('M99','M91','M92'),('M100','M95','M96')]
SF=[('M101','M97','M98'),('M102','M99','M100')]
FINAL=('M104','M101','M102')

def assign_thirds(third_groups, third_team):
    """Bijection: each qualifying third-place group -> a slot whose pool allows it."""
    slots=list(THIRD_SLOTS.items())
    order=sorted(third_groups, key=lambda g:sum(g in pool for _,(_,pool) in slots))  # most-constrained first
    assign={}
    def bt(i):
        if i==len(order): return True
        g=order[i]
        for m,(_,pool) in slots:
            if m not in assign and g in pool:
                assign[m]=g
                if bt(i+1): return True
                del assign[m]
        return False
    bt(0)
    return {m:third_team[g] for m,g in assign.items()}

def sim_tournament(R, record=False):
    W={};RU={};thirds=[]
    for g,t in GROUPS.items():
        rk,st=sim_group(t,R); W[g]=rk[0]; RU[g]=rk[1]
        thirds.append((g,rk[2],st[rk[2]]))
    thirds.sort(key=lambda z:(z[2][0],z[2][1],z[2][2],RNG.random()),reverse=True)
    best=thirds[:8]
    third_groups=[g for g,_,_ in best]; third_team={g:t for g,t,_ in best}
    third_for=assign_thirds(third_groups, third_team)
    res={}
    for m,(wg,pool) in THIRD_SLOTS.items():
        res[m]=ko(W[wg], third_for[m], R)
    for m,(p1,g1,p2,g2) in FIXED_R32.items():
        a=W[g1] if p1=='W' else RU[g1]; b=W[g2] if p2=='W' else RU[g2]
        res[m]=ko(a,b,R)
    for stage in (R16,QF,SF):
        for m,x,y in stage:
            res[m]=ko(res[x],res[y],R)
    champ=ko(res[FINAL[1]],res[FINAL[2]],R)
    return champ

if __name__=='__main__':
    df=pd.read_csv('results.csv'); df['date']=pd.to_datetime(df['date'])
    R=build_elo(df)
    N=20000; c=defaultdict(int)
    for _ in range(N): c[sim_tournament(R)]+=1
    print(f'=== REAL FIFA BRACKET ({N:,} sims) ===')
    for t,n in sorted(c.items(),key=lambda x:-x[1])[:16]:
        print(f'  {t:18s} {100*n/N:5.1f}%')