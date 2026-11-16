"""Aggregate held-out classification predictions without exporting participant rows."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external'))
from metrics import PersonBootstrap

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--plan',type=Path,nargs='+',required=True);p.add_argument('--training',type=Path,nargs='+',required=True);p.add_argument('--spectral',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    y=np.load(a.input/'labels.npy');jobs={};pairs=[]
    for path in a.plan:
        plan=json.loads(path.read_text());pairs=plan.get('cohort_metadata',{}).get('matching_pairs',pairs)
        for job in plan['training_jobs']:
            if job['id'] in jobs:assert jobs[job['id']]==job
            jobs[job['id']]=job
    groups={};expected={};unavailable={}
    for job in jobs.values():
        endpoints=[10,20,30,60,100] if job['mode']=='long100' else ['final']
        prefix='/'.join([job['mode'],job['architecture'],job['arm'],'seed'+str(job['seed']),'split'+str(job['split_seed'])])
        paths=[d/(job['id']+'.json') for d in a.training if (d/(job['id']+'.json')).exists()]
        assert len(paths)<=1,'Duplicate model outputs across input directories'
        result=json.loads(paths[0].read_text()) if paths else None
        if result is not None:assert result['job']==job and result['test_ids']==job['test']
        for endpoint in endpoints:
            key=prefix+'/'+str(endpoint);expected.setdefault(key,set()).update(job['test'])
            if result is None:continue
            epoch=str(result['epochs'] if endpoint=='final' else endpoint);values=groups.setdefault(key,{})
            for i,v in zip(result['test_ids'],expit(result['logits_by_epoch'][epoch])):
                assert i not in values;values[i]=float(v)
    predictions={}
    for key,ids in expected.items():
        if set(groups.get(key,{}))!=ids:unavailable[key]={'observed_n':len(groups.get(key,{})),'expected_n':len(ids)};continue
        ii=sorted(ids);predictions[key]=(ii,np.array([groups[key][i] for i in ii]))
    # Average probabilities only over the same held-out people and all three initializations.
    for key in list(predictions):
        if '/seed1109/' not in key:continue
        keys=[key.replace('/seed1109/','/seed'+str(s)+'/') for s in [1109,2209,3309]]
        if all(k in predictions for k in keys):
            ids=predictions[key][0];assert all(predictions[k][0]==ids for k in keys)
            predictions[key.replace('/seed1109/','/ensemble3/')]=(ids,np.mean([predictions[k][1] for k in keys],axis=0))
    if a.spectral:
        with np.load(a.spectral) as archive:
            for key in archive.files:
                q=archive[key];ii=np.flatnonzero(np.isfinite(q));predictions['spectral/'+key]=(ii.tolist(),q[ii])
    def pair_samples(ids,q):
        index={v:k for k,v in enumerate(ids)};pair=np.array([[index[x],index[z]] for x,z in pairs]);rng=np.random.default_rng(20261110);auc=[];ba=[]
        for _ in range(2000):
            ii=pair[rng.integers(0,len(pair),len(pair))].ravel();target=y[np.array(ids)[ii]];values=q[ii]
            auc.append(roc_auc_score(target,values));ba.append(.5*((values[target==1]>=.5).mean()+(values[target==0]<.5).mean()))
        return np.array(auc),np.array(ba)
    metrics={};contrasts={}
    for key,(ids,q) in predictions.items():
        yy=y[ids];b=PersonBootstrap(yy);result=b.evaluate(q);result.update(n=len(ids),n_adhd=int(yy.sum()),average_precision=float(average_precision_score(yy,q)),sensitivity=float((q[yy==1]>=.5).mean()),specificity=float((q[yy==0]<.5).mean()),bootstrap_unit='participant stratified by diagnosis')
        if key.startswith(('common/','duration/')):
            pos=np.flatnonzero(yy==1);neg=np.flatnonzero(yy==0);rng=np.random.default_rng(20261110)
            pc=rng.multinomial(len(pos),np.ones(len(pos))/len(pos),size=2000);nc=rng.multinomial(len(neg),np.ones(len(neg))/len(neg),size=2000)
            wins=(q[pos,None]>q[neg]).astype(float)+.5*(q[pos,None]==q[neg]);samples=np.einsum('bi,ij,bj->b',pc,wins,nc,optimize=True)/(len(pos)*len(neg));result['auc_ci95']=np.quantile(samples,[.025,.975]).tolist()
        if 'age_sex_matched' in key:
            auc,ba=pair_samples(ids,q);result.update(auc_ci95=np.percentile(auc,[2.5,97.5]).tolist(),ba_ci95=np.percentile(ba,[2.5,97.5]).tolist(),bootstrap_unit='matched pair')
        metrics[key]=result
        other=key.replace('/linked_basic/','/average_basic/').replace('/linked_eog/','/average_eog/').replace('_linked_basic','_average_basic').replace('_linked_eog','_average_eog')
        if other!=key and other in predictions:
            jj,z=predictions[other];assert ids==jj
            if 'age_sex_matched' in key:
                delta=pair_samples(ids,q)[0]-pair_samples(ids,z)[0];contrast={'difference':float(roc_auc_score(yy,q)-roc_auc_score(yy,z)),'ci95':np.percentile(delta,[2.5,97.5]).tolist()}
            elif key.startswith('duration/'):
                rng=np.random.default_rng(20261110);pos=np.flatnonzero(yy==1);neg=np.flatnonzero(yy==0);delta=[]
                for _ in range(2000):
                    ii=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))];delta.append(roc_auc_score(yy[ii],q[ii])-roc_auc_score(yy[ii],z[ii]))
                contrast={'difference':float(roc_auc_score(yy,q)-roc_auc_score(yy,z)),'ci95':np.quantile(delta,[.025,.975]).tolist()}
            else:contrast=b.auc_difference(q,z)
            contrasts[key+' minus '+other]=contrast
    for first,second in [('high_two_bands','low_four_bands'),('all_six_bands','low_four_bands')]:
        ka,kb='spectral/'+first,'spectral/'+second
        if ka in predictions and kb in predictions:
            ids,q=predictions[ka];jj,z=predictions[kb];assert ids==jj;contrasts[ka+' minus '+kb]=PersonBootstrap(y[ids]).auc_difference(q,z)
    a.output.mkdir(parents=True,exist_ok=True)
    aggregate={'classification':{'metrics':metrics,'unavailable':unavailable},'paired_classification_contrasts':{'reference':contrasts},'bootstrap':{'replicates':2000,'seed':20261110,'conditioning':'fixed fitted models'}}
    (a.output/'classification.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False));print('Wrote',len(metrics),'aggregate classifier summaries')
if __name__=='__main__':main()
