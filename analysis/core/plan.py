"""Construct reported training comparisons without inspecting predictions."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modules',nargs='+',choices=['core','stopping','long','partitions','matched','quality','duration','common','laplacian','label_null'],default=['core'])
    a=p.parse_args();c=a.input;y=np.load(c/'labels.npy');folds=json.loads((c/'folds.json').read_text());rows=json.loads((c/'cohort.json').read_text());arms=json.loads((c/'manifest.json').read_text())['arms'];jobs=[];metadata={}
    def add(mode,architecture,seeds,ff,epochs,use_arms=arms,split=20261109):
        for arm in use_arms:
            for seed in seeds:
                for f in ff:
                    assert not set(f['train'])&set(f['test']) and len(set(y[f['train']]))==2 and len(set(y[f['test']]))==2
                    jobs.append({'id':f'{mode}_{architecture}_{arm}_seed{seed}_split{split}_fold{f["fold"]}','mode':mode,'architecture':architecture,'arm':arm,'seed':seed,'split_seed':split,'epochs':epochs,**f})
    def restrict(ids):
        return [{'fold':f['fold'],'train':sorted(set(f['train'])&set(ids)),'test':sorted(set(f['test'])&set(ids))} for f in folds]
    for module in a.modules:
        if module=='core':
            for architecture in ['eegnet','temporal_cnn']:add('fixed30',architecture,[1109,2209,3309],folds,30)
        elif module=='stopping':add('balanced_stop','eegnet',[1109,2209,3309],folds,30)
        elif module=='long':add('long100','eegnet',[1109],folds,100)
        elif module=='label_null':add('label_null','eegnet',[1109],folds,30)
        elif module=='partitions':
            for split in [20261110,20261111]:
                ff=[{'fold':k,'train':tr.tolist(),'test':te.tolist()} for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=split).split(y,y))]
                add('split30','eegnet',[1109],ff,30,split=split)
        elif module=='quality':
            ids=np.flatnonzero(np.load(c/'quality_features.npy')[:,1]<=.10);ff=restrict(ids)
            assert all(min(np.bincount(y[f['test']],minlength=2))>=2 and min(np.bincount(y[f['train']],minlength=2))>=10 for f in ff)
            metadata['quality_indices']=ids.tolist();add('strict_quality','eegnet',[1109],ff,30)
        elif module=='matched':
            dem=np.array([[float(r['age_years']),float(r['sex']=='Male')] for r in rows]);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
            age=np.abs(dem[pos,0,None]-dem[neg,0][None]);same=dem[pos,1,None]==dem[neg,1][None]
            cost=np.full((len(pos),len(neg)+len(pos)),10000.);cost[:,:len(neg)]=np.where(same&(age<=3),age,100000.)
            rr,cc=linear_sum_assignment(cost);pairs=sorted([[int(pos[r]),int(neg[j])] for r,j in zip(rr,cc) if j<len(neg) and cost[r,j]<10000]);assert len(pairs)>=20
            groups=np.array_split(np.random.default_rng(20261109).permutation(len(pairs)),5);ids=sorted(i for pair in pairs for i in pair);ff=[]
            for k,group in enumerate(groups):
                te=sorted(i for j in group for i in pairs[j]);ff.append({'fold':k,'train':sorted(set(ids)-set(te)),'test':te})
            metadata['matching_pairs']=pairs;add('age_sex_matched','eegnet',[1109],ff,30)
        elif module=='duration':
            # Reported exact-duration sensitivity, selected using metadata counts.
            durations=np.array([float(r['ec_duration_s']) for r in rows]);candidates=[v for v in sorted(set(durations)) if len(set(y[durations==v]))==2]
            duration=sorted(candidates,key=lambda v:(-int((durations==v).sum()),str(v)))[0];assert duration==121
            ids=np.flatnonzero(durations==duration);metadata['duration_seconds']=float(duration);metadata['duration_indices']=ids.tolist()
            add('duration','eegnet',[1109,2209,3309],restrict(ids),30,use_arms=['linked_basic','average_basic'])
        elif module=='common':
            linked=np.load(c/'linked_basic.npy');average=np.load(c/'average_basic.npy');common=linked.mean(2,keepdims=True)
            assert np.allclose(linked-common,average,rtol=1e-5,atol=5e-5)
            np.save(c/'common_only.npy',np.repeat(common,19,axis=2));np.save(c/'common_removed.npy',average)
            add('common','eegnet',[1109,2209,3309],folds,30,use_arms=['common_only','common_removed'])
        elif module=='laplacian':
            import mne
            channels=json.loads((c/'manifest.json').read_text())['channels'];info=mne.create_info(channels,200,'eeg');info.set_montage(mne.channels.make_standard_montage('standard_1020'))
            raw=mne.io.RawArray(np.eye(19),info,verbose='ERROR');matrix=mne.preprocessing.compute_current_source_density(raw,sphere='auto',lambda2=1e-5,stiffness=4,n_legendre_terms=50,copy=True).get_data()
            sv=np.linalg.svd(matrix,compute_uv=False);assert np.isfinite(matrix).all() and sv[0]/sv[-2]<1/np.finfo(float).eps;np.save(c/'csd_matrix.npy',matrix)
            for suffix in ['basic','eog']:
                linked=np.load(c/('linked_'+suffix+'.npy'));average=np.load(c/('average_'+suffix+'.npy'))
                x=np.einsum('cd,nwdt->nwct',matrix,linked.astype(np.float64)*1e-6).astype(np.float32);z=np.einsum('cd,nwdt->nwct',matrix,average.astype(np.float64)*1e-6)
                assert np.linalg.norm(x-z)/max(np.linalg.norm(z),1e-12)<1e-4;np.save(c/('csd_'+suffix+'.npy'),x)
            add('csd','eegnet',[1109],folds,30,use_arms=['csd_basic','csd_eog'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'training_jobs':jobs,'cohort_metadata':metadata},indent=2));print('Created',len(jobs),'model jobs; no training performed')
if __name__=='__main__':main()
