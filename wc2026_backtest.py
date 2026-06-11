"""
Out-of-sample validation for the WC2026 model (prequential / walk-forward).

Marches through every international match in date order, maintaining Elo. For
each match in the held-out window it predicts win/draw/loss BEFORE seeing the
result (using only prior data), then scores it. This is honest out-of-sample
evaluation: no result is ever used to predict itself.

Reports: model vs base-rate baseline (log-loss + accuracy), and whether the
confederation calibration improves things on the matches it actually affects.
"""
import pandas as pd, numpy as np
from math import exp, factorial
from collections import defaultdict, Counter

TEST_START = pd.Timestamp('2023-01-01')
CONTINENTAL = {'Copa América','UEFA Euro','African Cup of Nations','AFC Asian Cup','Gold Cup','Oceania Nations Cup'}
CONF_TOURNEYS = {'CONMEBOL':['Copa América'],'UEFA':['UEFA Euro','UEFA Nations League','UEFA Euro qualification'],
  'AFC':['AFC Asian Cup','AFC Asian Cup qualification'],'CAF':['African Cup of Nations','African Cup of Nations qualification'],
  'CONCACAF':['Gold Cup','CONCACAF Nations League'],'OFC':['Oceania Nations Cup']}

def k_of(t):                       # mirrors wc2026_bracket.build_elo
    t=str(t)
    if 'World Cup' in t and 'qualification' not in t: return 60
    if t in CONTINENTAL: return 50
    if 'qualification' in t or 'Nations League' in t: return 40
    if 'Friendly' in t: return 20
    return 30

def pois(l): return [exp(-l)*l**k/factorial(k) for k in range(11)]
def outcome_probs(Ra, Rb, neutral):
    ha=0 if neutral else 65; d=(Ra-Rb+ha)/400
    pa=pois(1.36*np.exp(0.45*d)); pb=pois(1.36*np.exp(-0.45*d))
    ph=pdr=paw=0.0
    for x in range(11):
        for y in range(11):
            p=pa[x]*pb[y]
            if x>y: ph+=p
            elif x==y: pdr+=p
            else: paw+=p
    s=ph+pdr+paw; return [ph/s, pdr/s, paw/s]

def confed_map(df):
    cnt=defaultdict(Counter)
    for c,ts in CONF_TOURNEYS.items():
        sub=df[df['tournament'].isin(ts)]
        for t in pd.concat([sub['home_team'],sub['away_team']]).dropna(): cnt[t][c]+=1
    return {t:c.most_common(1)[0][0] for t,c in cnt.items()}

def fit_offsets(played, conf):     # fit on pre-TEST inter-confederation matches only (no leakage)
    Rpre={}
    for r in played[played.date<TEST_START].itertuples():
        Ra=Rpre.get(r.home_team,1500);Rb=Rpre.get(r.away_team,1500);ha=0 if r.neutral else 65
        ea=1/(1+10**(-(Ra-Rb+ha)/400));gd=abs(r.home_score-r.away_score)
        mult=1 if gd<=1 else (1.5 if gd==2 else 1.75+(gd-3)/8)
        sa=1 if r.home_score>r.away_score else (0 if r.home_score<r.away_score else .5)
        K=k_of(r.tournament)*mult; Rpre[r.home_team]=Ra+K*(sa-ea); Rpre[r.away_team]=Rb-K*(sa-ea)
    inter=[]
    for r in played[(played.date>='2014-01-01')&(played.date<TEST_START)].itertuples():
        ca,cb=conf.get(r.home_team),conf.get(r.away_team)
        if ca and cb and ca!=cb:
            ha=0 if r.neutral else 65; act=1 if r.home_score>r.away_score else (0 if r.home_score<r.away_score else .5)
            inter.append((r.home_team,r.away_team,ca,cb,ha,act))
    off={c:0.0 for c in CONF_TOURNEYS}
    for _ in range(200):
        num=defaultdict(float); den=defaultdict(int)
        for a,b,ca,cb,ha,act in inter:
            e=1/(1+10**(-((Rpre.get(a,1500)+off[ca])-(Rpre.get(b,1500)+off[cb])+ha)/400))
            num[ca]+=act-e; den[ca]+=1; num[cb]+=(1-act)-(1-e); den[cb]+=1
        for c in off:
            if den[c]: off[c]+=(num[c]/den[c])/0.00144*0.5
        mn=np.mean(list(off.values()))
        for c in off: off[c]-=mn
    return off

def main():
    df=pd.read_csv('results.csv'); df['date']=pd.to_datetime(df['date'])
    played=df.dropna(subset=['home_score','away_score']).sort_values('date')
    conf=confed_map(df); off=fit_offsets(played, conf)
    tr=played[played.date<TEST_START]
    base=[ (tr.home_score>tr.away_score).mean(), (tr.home_score==tr.away_score).mean(), (tr.home_score<tr.away_score).mean() ]
    barg=int(np.argmax(base)); eps=1e-15
    R={}; mll=[];bll=[];macc=0;bacc=0;n=0; iu=[];ic=[];ni=0
    for r in played.itertuples():
        a,b=r.home_team,r.away_team; Ra=R.get(a,1500); Rb=R.get(b,1500); ca,cb=conf.get(a),conf.get(b)
        if r.date>=TEST_START:
            out=0 if r.home_score>r.away_score else (1 if r.home_score==r.away_score else 2)
            p=outcome_probs(Ra,Rb,r.neutral)
            mll.append(-np.log(max(p[out],eps))); bll.append(-np.log(max(base[out],eps)))
            macc+=int(np.argmax(p)==out); bacc+=int(barg==out); n+=1
            if ca and cb and ca!=cb:                     # calibration only acts here
                pc=outcome_probs(Ra+off.get(ca,0),Rb+off.get(cb,0),r.neutral)
                iu.append(-np.log(max(p[out],eps))); ic.append(-np.log(max(pc[out],eps))); ni+=1
        ha=0 if r.neutral else 65; ea=1/(1+10**(-(Ra-Rb+ha)/400)); gd=abs(r.home_score-r.away_score)
        mult=1 if gd<=1 else (1.5 if gd==2 else 1.75+(gd-3)/8)
        sa=1 if r.home_score>r.away_score else (0 if r.home_score<r.away_score else .5)
        K=k_of(r.tournament)*mult; R[a]=Ra+K*(sa-ea); R[b]=Rb-K*(sa-ea)
    print(f'Out-of-sample backtest, {n:,} held-out matches since {TEST_START.date()}\n')
    print(f'  MODEL     log-loss {np.mean(mll):.4f}   accuracy {100*macc/n:.1f}%')
    print(f'  BASELINE  log-loss {np.mean(bll):.4f}   accuracy {100*bacc/n:.1f}%   (predict base rates)')
    print(f'  -> {100*(np.mean(bll)-np.mean(mll))/np.mean(bll):.1f}% lower log-loss than baseline\n')
    print(f'  Confederation calibration, on the {ni:,} inter-confederation holdout matches it affects:')
    print(f'    uncalibrated {np.mean(iu):.4f}  ->  calibrated {np.mean(ic):.4f}  ({100*(np.mean(iu)-np.mean(ic))/np.mean(iu):+.1f}%)')

if __name__=='__main__':
    main()
