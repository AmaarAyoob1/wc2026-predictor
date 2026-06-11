"""
WC2026 predictor: full-history Elo + recent-form residual layer + real FIFA bracket.
Form layer (lambda=500): nudges each team by how its last 6 years compared to what its
own Elo predicted at the time (recency-weighted, competitive matches count more, shrunk
for small samples). Keeps full-history separation; adds recency as a correction, not a reset.
"""
import numpy as np, pandas as pd, json
from collections import defaultdict

REF=pd.Timestamp('2026-06-11'); WINDOW=REF-pd.Timedelta(days=int(365.25*6)); HALF=2.0; LAM=500; SHRINK=2.0
def k_of(t):
    t=str(t)
    if 'World Cup' in t and 'qualification' not in t: return 60
    if 'qualification' in t or 'Nations League' in t: return 40
    if 'Friendly' in t: return 20
    return 30
def elo_and_form(data,base=1500,home_adv=65):
    R=defaultdict(lambda:base);rec=defaultdict(list)
    for r in data.sort_values('date').itertuples():
        ha=0 if r.neutral else home_adv
        e=1/(1+10**(-(R[r.home_team]-R[r.away_team]+ha)/400))
        s=1 if r.home_score>r.away_score else (0 if r.home_score<r.away_score else .5)
        gd=abs(r.home_score-r.away_score);mult=1 if gd<=1 else (1.5 if gd==2 else 1.75+(gd-3)/8)
        if r.date>=WINDOW:
            imp=k_of(r.tournament)/40.0
            rec[r.home_team].append((r.date,s-e,imp));rec[r.away_team].append((r.date,-(s-e),imp))
        K=k_of(r.tournament)*mult;R[r.home_team]+=K*(s-e);R[r.away_team]-=K*(s-e)
    return dict(R),rec
def form_layer(rec):
    out={}
    for t,lst in rec.items():
        num=den=0.0
        for d,res,imp in lst:
            w=(0.5**(((REF-d).days/365.25)/HALF))*imp;num+=w*res;den+=w
        out[t]=LAM*num/(den+SHRINK)
    return out

GROUPS={'A':['Mexico','South Korea','Czech Republic','South Africa'],'B':['Switzerland','Canada','Bosnia and Herzegovina','Qatar'],'C':['United States','Paraguay','Australia','Turkey'],'D':['Brazil','Morocco','Scotland','Haiti'],'E':['Germany','Ecuador','Ivory Coast','Curaçao'],'F':['Netherlands','Japan','Sweden','Tunisia'],'G':['Belgium','Egypt','Iran','New Zealand'],'H':['Spain','Uruguay','Saudi Arabia','Cape Verde'],'I':['France','Senegal','Norway','Iraq'],'J':['Argentina','Austria','Algeria','Jordan'],'K':['Portugal','Colombia','Uzbekistan','DR Congo'],'L':['England','Croatia','Ghana','Panama']}
ALL=sum(GROUPS.values(),[])
RNG=np.random.default_rng(42);BASE,BETA=1.36,0.45
def sm(a,b,R):
    d=(R.get(a,1500)-R.get(b,1500))/400;return RNG.poisson(BASE*np.exp(BETA*d)),RNG.poisson(BASE*np.exp(-BETA*d))
def ko(a,b,R):
    x,y=sm(a,b,R)
    if x>y:return a
    if y>x:return b
    return a if RNG.random()<1/(1+10**(-(R.get(a,1500)-R.get(b,1500))/400)) else b
def sg(teams,R):
    pts=defaultdict(int);gf=defaultdict(int);ga=defaultdict(int)
    for i in range(4):
        for j in range(i+1,4):
            a,b=teams[i],teams[j];x,y=sm(a,b,R);gf[a]+=x;ga[a]+=y;gf[b]+=y;ga[b]+=x
            if x>y:pts[a]+=3
            elif y>x:pts[b]+=3
            else:pts[a]+=1;pts[b]+=1
    rk=sorted(teams,key=lambda t:(pts[t],gf[t]-ga[t],gf[t],RNG.random()),reverse=True)
    return rk,{t:(pts[t],gf[t]-ga[t],gf[t]) for t in teams}
TS={'M74':('E',set('ABCDF')),'M77':('I',set('CDFGH')),'M79':('A',set('CEFHI')),'M80':('L',set('EHIJK')),'M81':('D',set('BEFIJ')),'M82':('G',set('AEHIJ')),'M85':('B',set('EFGIJ')),'M87':('K',set('DEIJL'))}
FX={'M73':('RU','A','RU','B'),'M75':('W','F','RU','C'),'M76':('W','C','RU','F'),'M78':('RU','E','RU','I'),'M83':('RU','K','RU','L'),'M84':('W','H','RU','J'),'M86':('W','J','RU','H'),'M88':('RU','D','RU','G')}
R16=[('M89','M74','M77'),('M90','M73','M75'),('M91','M76','M78'),('M92','M79','M80'),('M93','M83','M84'),('M94','M81','M82'),('M95','M86','M88'),('M96','M85','M87')]
QF=[('M97','M89','M90'),('M98','M93','M94'),('M99','M91','M92'),('M100','M95','M96')];SF=[('M101','M97','M98'),('M102','M99','M100')];FN=('M104','M101','M102')
def at(tg,tt):
    slots=list(TS.items());order=sorted(tg,key=lambda g:sum(g in pool for _,(_,pool) in slots));asg={}
    def bt(i):
        if i==len(order):return True
        g=order[i]
        for m,(_,pool) in slots:
            if m not in asg and g in pool:
                asg[m]=g
                if bt(i+1):return True
                del asg[m]
        return False
    bt(0);return {m:tt[g] for m,g in asg.items()}
def sim(R):
    W={};RU={};th=[]
    for g,t in GROUPS.items():
        rk,st=sg(t,R);W[g]=rk[0];RU[g]=rk[1];th.append((g,rk[2],st[rk[2]]))
    th.sort(key=lambda z:(z[2][0],z[2][1],z[2][2],RNG.random()),reverse=True);best=th[:8]
    tf=at([g for g,_,_ in best],{g:t for g,t,_ in best});res={}
    for m,(wg,pool) in TS.items():res[m]=ko(W[wg],tf[m],R)
    for m,(p1,g1,p2,g2) in FX.items():
        a=W[g1] if p1=='W' else RU[g1];b=W[g2] if p2=='W' else RU[g2];res[m]=ko(a,b,R)
    for stg in (R16,QF,SF):
        for m,x,y in stg:res[m]=ko(res[x],res[y],R)
    return ko(res[FN[1]],res[FN[2]],R)

CONFED={
 "Mexico":"CONCACAF","South Korea":"AFC","Czech Republic":"UEFA","South Africa":"CAF",
 "Switzerland":"UEFA","Canada":"CONCACAF","Bosnia and Herzegovina":"UEFA","Qatar":"AFC",
 "United States":"CONCACAF","Paraguay":"CONMEBOL","Australia":"AFC","Turkey":"UEFA",
 "Brazil":"CONMEBOL","Morocco":"CAF","Scotland":"UEFA","Haiti":"CONCACAF",
 "Germany":"UEFA","Ecuador":"CONMEBOL","Ivory Coast":"CAF","Curaçao":"CONCACAF",
 "Netherlands":"UEFA","Japan":"AFC","Sweden":"UEFA","Tunisia":"CAF",
 "Belgium":"UEFA","Egypt":"CAF","Iran":"AFC","New Zealand":"OFC",
 "Spain":"UEFA","Uruguay":"CONMEBOL","Saudi Arabia":"AFC","Cape Verde":"CAF",
 "France":"UEFA","Senegal":"CAF","Norway":"UEFA","Iraq":"AFC",
 "Argentina":"CONMEBOL","Austria":"UEFA","Algeria":"CAF","Jordan":"AFC",
 "Portugal":"UEFA","Colombia":"CONMEBOL","Uzbekistan":"AFC","DR Congo":"CAF",
 "England":"UEFA","Croatia":"UEFA","Ghana":"CAF","Panama":"CONCACAF"}
OFFSETS={"UEFA":82,"CONMEBOL":61,"CAF":67,"AFC":6,"CONCACAF":-4,"OFC":-211}
import numpy as _np
VALUES={'Mexico':200,'South Korea':190,'Czech Republic':260,'South Africa':70,'Switzerland':320,'Canada':270,'Bosnia and Herzegovina':150,'Qatar':35,'United States':360,'Paraguay':90,'Australia':90,'Turkey':450,'Brazil':1135,'Morocco':360,'Scotland':220,'Haiti':30,'Germany':800,'Ecuador':290,'Ivory Coast':260,'Curaçao':45,'Netherlands':660,'Japan':250,'Sweden':330,'Tunisia':70,'Belgium':460,'Egypt':160,'Iran':90,'New Zealand':35,'Spain':900,'Uruguay':400,'Saudi Arabia':35,'Cape Verde':70,'France':1195,'Senegal':360,'Norway':520,'Iraq':35,'Argentina':620,'Austria':260,'Algeria':210,'Jordan':30,'Portugal':1000,'Colombia':320,'Uzbekistan':45,'DR Congo':130,'England':1345,'Croatia':360,'Ghana':160,'Panama':35}
LAMV=45

if __name__=='__main__':
    df=pd.read_csv('results.csv');df['date']=pd.to_datetime(df['date'])
    played=df.dropna(subset=['home_score','away_score'])
    R_full,rec=elo_and_form(played)
    form=form_layer(rec)
    _mln=_np.mean([_np.log(VALUES[t]) for t in ALL])
    val={t:LAMV*(_np.log(VALUES[t])-_mln) for t in ALL}
    R={t:R_full.get(t,1500)+OFFSETS[CONFED[t]]+form.get(t,0.0)+val[t] for t in ALL}
    N=20000;c=defaultdict(int)
    for _ in range(N):c[sim(R)]+=1
    print(f"=== LOCKED BOARD: Elo + confed + form(500) + squad-value(45), {N:,} sims) ===")
    for t,n in sorted(c.items(),key=lambda x:-x[1])[:16]:
        print(f"  {t:18s} {100*n/N:5.1f}%")
    # export ratings for the dashboard
    out={'groups':GROUPS,'model':'elo+confed+form+value','ratings':{t:round(R[t],1) for t in ALL},'base':BASE,'beta':BETA,'lambda_form':LAM,'lambda_value':LAMV}
    json.dump(out,open('wc2026_ratings_final.json','w'),ensure_ascii=False)
    print("\nWrote wc2026_ratings_final.json")
