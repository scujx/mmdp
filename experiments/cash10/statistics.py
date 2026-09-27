"""Predeclared whole-training-seed inference for the unified Search matrix."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import itertools,json,hashlib
import numpy as np
from pathlib import Path

FAMILIES=('flat_ppo','mmdp_ppo','flat_aa_ppo','mmdp_aa_ppo')
MODES=(0,576,1152,2304)
NBOOT=100000
RNG=2026091501
TOL=1e-12


def bootstrap_indices():
    return np.random.default_rng(RNG).integers(0,10,size=(NBOOT,10),dtype=np.int64)


def holm(p):
    p=np.asarray(p,float);order=np.argsort(p,kind='stable');out=np.empty(len(p));current=0.
    for rank,i in enumerate(order):
        current=max(current,min(1.,(len(p)-rank)*p[i]));out[i]=current
    return out


def contrast_specs():
    # Vectors refer to family x budget, with budget index0 = no Search.
    items=[]
    def add(name,label,terms,primary=False):
        w=np.zeros((4,4))
        for family,mode,coefficient in terms:w[family,MODES.index(mode)]+=coefficient
        items.append(dict(name=name,label=label,primary=primary,weights=w.tolist()))
    add('H1','MMDP-AA Search minus Flat-AA Search at2304',[(3,2304,1),(2,2304,-1)],True)
    add('H2','MMDP-AA Search minus MMDP-AA Raw at2304',[(3,2304,1),(3,0,-1)],True)
    for cap in MODES[1:]:
        for f,label in enumerate(FAMILIES):
            if f==3 and cap==2304:continue
            add(f'gain_{label}_{cap}',f'{label} Search minus Raw at{cap}',[(f,cap,1),(f,0,-1)])
        add(f'nonAA_interface_{cap}',f'MMDP minus Flat Search at{cap}',[(1,cap,1),(0,cap,-1)])
        if cap!=2304:
            add(f'AA_interface_{cap}',f'MMDP-AA minus Flat-AA Search at{cap}',[(3,cap,1),(2,cap,-1)])
        add(f'AA_difference_in_differences_{cap}',f'AA Search interface gap minus AA Raw interface gap at{cap}',[(3,cap,1),(2,cap,-1),(3,0,-1),(2,0,1)])
    assert len(items)==21 and sum(x['primary'] for x in items)==2
    return items


def summarize_gap(gap,indices,primary=False):
    gap=np.asarray(gap,dtype=np.float64);assert gap.shape==(10,) and np.isfinite(gap).all()
    mean=float(gap.mean());sd=0. if np.all(gap==gap[0]) else float(gap.std(ddof=1));samples=gap[indices].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975],method='linear')
    out=dict(gaps=gap.tolist(),mean=mean,sd=sd,ci_low=float(lo),ci_high=float(hi),positive_seeds=int((gap>0).sum()),dz=mean/sd if sd!=0 else None,dz_status='finite' if sd else 'undefined_zero_variance' if mean==0 else 'positive_infinity' if mean>0 else 'negative_infinity')
    if primary:
        signs=np.asarray(list(itertools.product((-1.,1.),repeat=10)))
        null=(signs*gap).mean(axis=1)
        out.update(exact_p=float(np.mean(np.abs(null)>=abs(mean)-TOL)),exact_distribution=null.tolist())
    return out,samples


def calculate(endpoints):
    x=np.asarray(endpoints,np.float64);assert x.shape==(10,4,4) and np.isfinite(x).all()
    idx=bootstrap_indices();results=[];samples=[]
    for spec in contrast_specs():
        gaps=(x*np.asarray(spec['weights'])).sum(axis=(1,2))
        summary,bs=summarize_gap(gaps,idx,spec['primary'])
        results.append(dict(**spec,**summary));samples.append(bs)
    for row,p in zip(results[:2],holm([z['exact_p'] for z in results[:2]])):row['holm_p']=float(p)
    budgets=[]
    for f,label in enumerate(FAMILIES):
        for j,cap in enumerate(MODES):
            z,_=summarize_gap(x[:,f,j],idx)
            budgets.append(dict(family=label,cap=cap,means=x[:,f,j].tolist(),mean=z['mean'],sd=z['sd'],ci_low=z['ci_low'],ci_high=z['ci_high']))
    result=dict(primary=results[:2],secondary=results[2:],budget_means=budgets,endpoint_matrix=x.tolist(),families=FAMILIES,budgets=MODES,bootstrap_count=NBOOT,bootstrap_rng=RNG,bootstrap_indices_sha256=hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(),exact_tolerance=TOL,ci='Marginal unadjusted paired percentile95%; whole training seeds',holm_family=['H1','H2'])
    return result,idx,np.asarray(samples)


def selftest():
    checks=[];idx=bootstrap_indices()
    for label,gap,expected in [('zeros',np.zeros(10),1.),('constant_positive',np.ones(10),2/1024),('constant_negative',-np.ones(10),2/1024),('tiny_constant',np.full(10,1e-14),1.),('symmetric',np.arange(-5,5)+.5,1.)]:
        r,_=summarize_gap(gap,idx,True);assert r['exact_p']==expected
        assert r['mean']==gap.mean() and r['positive_seeds']==int((gap>0).sum());checks.append(label)
        if np.all(gap==gap[0]):assert r['sd']==0 and r['dz'] is None
    assert np.allclose(holm([.04,.01]),[.04,.02]);assert np.allclose(holm([.7,.8]),[1,1]);checks.append('holm')
    x=np.arange(160,dtype=float).reshape(10,4,4)/10
    result,_,_=calculate(x)
    assert np.allclose(result['primary'][0]['gaps'],x[:,3,3]-x[:,2,3])
    assert np.allclose(result['primary'][1]['gaps'],x[:,3,3]-x[:,3,0])
    for cap in MODES[1:]:
        row=next(z for z in result['secondary'] if z['name']==f'AA_difference_in_differences_{cap}')
        j=MODES.index(cap)
        assert np.allclose(row['gaps'],(x[:,3,j]-x[:,2,j])-(x[:,3,0]-x[:,2,0]))
    checks.extend(['H1_pairing','H2_pairing','all_three_direct_DID','all_21_contrasts'])
    assert len(result['secondary'])==19 and len(result['budget_means'])==16
    return dict(passed=True,checks=checks,bootstrap_indices_sha256=result['bootstrap_indices_sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_returns_accessed=0)


if __name__=='__main__':
    result=selftest()
    print(json.dumps(result,indent=2))
