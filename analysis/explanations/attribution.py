"""Participant-mean expected IG and DeepSHAP with reference-aware perturbations."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch import nn
import shap
from scipy.special import expit

def evaluate(model,x):
    model.eval()
    with torch.no_grad(): return np.concatenate([model(b).cpu().numpy() for b in x.split(4)])

def project(x,average):
    return x-x.mean(dim=-2,keepdim=True) if average else x

def mask_effects(model,x,masks,average,original):
    logits=[];energies=[]
    denominator=float(x.square().sum().cpu())
    for start in range(0,len(masks),4):
        # Multiplicative mask avoids PyTorch 2.0 CUDA advanced-index assignment bug.
        # Construct indices on CPU; numerically identical to zeroing those channels.
        keep=np.ones((len(masks[start:start+4]),19),dtype=np.float32)
        for j,mask in enumerate(masks[start:start+4]): keep[j,mask]=0
        batch=x[None]*torch.as_tensor(keep,device=x.device,dtype=x.dtype)[:,None,:,None]
        batch=project(batch,average)
        logits.extend(evaluate(model,batch).tolist())
        energies.extend(((batch-x[None]).square().sum((1,2,3))/max(denominator,1e-12)).cpu().numpy().tolist())
    sign=1 if original>=0 else -1
    return sign*(original-np.asarray(logits)),np.asarray(energies)
class WindowTarget(nn.Module):
    def __init__(self,base,average):
        super().__init__();self.base=base;self.average=average
    def forward(self,x):
        if self.average:x=x-x.mean(2,keepdim=True)
        return self.base.window_logits(x)[:,None]

def expected_ig(wrapper,x,background,steps=32):
    # Each path is a participant bag of15 windows relative to one background window.
    attribution=torch.zeros_like(x)
    for reference in background:
        diff=x-reference
        g=torch.zeros_like(x)
        for start in range(0,steps,4):
            alpha=(torch.arange(start,min(start+4,steps),device=x.device,dtype=x.dtype)+.5)/steps
            z=(reference[None,None]+alpha[:,None,None,None,None]*diff[None]).requires_grad_(True)
            output=wrapper(z.reshape(-1,1,19,800)).reshape(len(alpha),len(x)).mean(1)
            grad=torch.autograd.grad(output.sum(),z)[0]
            g+=grad.sum(0)/steps
        attribution+=diff*g/len(background)
    return attribution.detach().cpu().numpy()[:,0]

def deep_shap(wrapper,x,explainer):
    values=explainer.shap_values(x,check_additivity=True)
    if isinstance(values,list):values=values[0]
    values=np.asarray(values)
    if values.ndim==5:values=values[...,0]
    assert values.shape==tuple(x.shape),values.shape
    return (values[:,0]/len(x)).astype('float32')
def measures(model,wrapper,x,bg,attr,arm,rois,i,steps):
    with torch.no_grad():
        logit=float(wrapper(x).mean().cpu());expected=float(wrapper(bg).mean().cpu())
        direct=float(model(x[:,0][None]).cpu())
    assert abs(logit-direct)<1e-4
    sign=1 if logit>=0 else -1;delta=logit-expected
    signed={r:float(attr[:,ix,:].sum()) for r,ix in rois.items()}
    scores={r:sign*v for r,v in signed.items()};top=max(scores,key=scores.get)
    rng=np.random.default_rng(91000+i)
    masks=[rois[top]]+[rng.choice(19,len(rois[top]),replace=False).tolist() for _ in range(100)]
    drops,energy=mask_effects(model,x[:,0],masks,'average' in arm,logit)
    nearest=np.argsort(np.abs(energy[1:]-energy[0]),kind='stable')[:10]+1
    probability_drops=expit(sign*logit)-expit(sign*logit-drops)
    faith=float(drops[0]-drops[1:].mean());res=float(attr.sum())-delta
    return {'logit':logit,'background_logit':expected,'target_difference':delta,
        'relative_residual':abs(res)/max(abs(delta),.01),'completeness_residual':res,
        'signed_roi_scores':signed,'roi_scores':scores,'top_region':top,
        'faithfulness':faith,'normalized_faithfulness':faith/max(abs(delta),.1),
        'probability_faithfulness':float(probability_drops[0]-probability_drops[1:].mean()),
        'energy_matched_faithfulness':float(drops[0]-drops[nearest].mean()),
        'energy_matched_probability_faithfulness':float(probability_drops[0]-probability_drops[nearest].mean()),
        'energy_match_absolute_gap':float(np.abs(energy[nearest]-energy[0]).mean()),
        'ig_steps':steps}

def perturbations(model,x,bg,attr,record,arm,i,rois):
    sign=1 if record['logit']>=0 else -1;average='average' in arm
    score=sign*attr.sum((0,2));order=np.argsort(-score,kind='stable')
    sizes=[1,3,5,10,19];orders=[order]
    rng=np.random.default_rng(92000+i)
    orders += [rng.permutation(19) for _ in range(100)]
    masks=[o[:n].tolist() for o in orders for n in sizes]
    drop,_=mask_effects(model,x[:,0],masks,average,record['logit'])
    curves=(expit(sign*record['logit'])-expit(sign*record['logit']-drop)).reshape(101,5)
    curves=np.c_[np.zeros(101),curves];fractions=np.array([0,*sizes])/19
    auc=np.trapz(curves,fractions,axis=1)
    # Identical donors and masks for each recipient; replacement uses normalized source units.
    top=rois[record['top_region']];rng=np.random.default_rng(91000+i)
    masks=[top]+[rng.choice(19,len(top),replace=False).tolist() for _ in range(100)]
    donor_scores=[]
    for donor in bg[:,0]:
        predictions=[]
        for offset in range(0,len(masks),4):
            part=masks[offset:offset+4]
            keep=np.ones((len(part),19),dtype='float32')
            for j,mask in enumerate(part):keep[j,mask]=0
            keep=torch.tensor(keep,device=x.device)[:,None,:,None]
            modified=project(x[:,0][None]*keep+donor[None,None]*(1-keep),average)
            predictions.extend(evaluate(model,modified).tolist())
        donor_scores.append(expit(sign*record['logit'])-expit(sign*np.asarray(predictions)))
    donor_scores=np.asarray(donor_scores)
    return {'deletion_fractions':fractions.tolist(),'top_probability_drop_curve':curves[0].tolist(),
        'random_mean_probability_drop_curve':curves[1:].mean(0).tolist(),
        'top_deletion_auc':float(auc[0]),'random_mean_deletion_auc':float(auc[1:].mean()),
        'deletion_auc_difference':float(auc[0]-auc[1:].mean()),
        'donor_probability_faithfulness':float((donor_scores[:,0]-donor_scores[:,1:].mean(1)).mean()),
        'donor_specific_faithfulness':(donor_scores[:,0]-donor_scores[:,1:].mean(1)).tolist()}
