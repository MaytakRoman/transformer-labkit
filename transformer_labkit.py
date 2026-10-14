"""Offline assignment generator for four transformer labs, 60 variants.

The generator supplies data and contracts, not completed student algorithms.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np

SCHEMA = '1.0'
VERSION = '1.0.0'
TASKS = ('copy', 'shift', 'marker', 'parity', 'associative', 'induction')
SPLITS = ('train', 'valid', 'test')

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def seed(lab, variant, role):
    # Explicit delimiter resolves the concatenation convention in formula 8.1.
    return int(sha(f'{lab}|{variant}|{role}|{SCHEMA}'.encode())[:16], 16) % 2**63

def rng(lab, variant, role):
    return np.random.default_rng(seed(lab, variant, role))

def parameters(variant, lab):
    if not 1 <= variant <= 60 or not 1 <= lab <= 4:
        raise ValueError('lab must be 1..4 and variant must be 1..60')
    v = variant - 1
    return dict(variant=variant, lab=lab, table_seed=1000+7919*variant,
                task=TASKS[v % 6], max_length=16+8*(v % 6),
                vocab_size=32+16*(v % 5), noise=0.05*(v % 4),
                heads=1+v % 4, head_dim=4+2*(v % 3),
                temperature=(0.5, 1., 2.)[v % 3],
                mask_type=('causal', 'bidirectional', 'window')[v % 3],
                model_width=16+8*(v % 4), learning_rate=(.001,.003,.01)[v % 3],
                layers=1+v % 3, aggregation=('mean','last')[v % 2],
                accuracy_min=.65+.05*(v % 4), ece_max=.10+.02*(v % 3),
                memory_max_mib=64+16*(v % 4), latency_max_ms=5+2*(v % 3))

def sample(cfg, split, index):
    """Create a clean rule and apply observation noise in an independent stream."""
    lab, v = cfg['lab'], cfg['variant']
    g = rng(lab,v,f'{split}/{index}/latent')
    role = f'{split}/{index}'
    task = cfg['task']; size=cfg['max_length']; vocab=cfg['vocab_size']
    n=int(g.integers(max(3,size//2),size+1)); x=g.integers(3,vocab,size=n).tolist()
    positions=[]; values=[]; latent={'task':task, 'length':n}
    if task == 'copy':
        m=(n-1)//2; payload=g.integers(3,vocab,size=m).tolist()
        x=payload+[1]+[2]*m; n=len(x)
        positions=list(range(m+1,n)); values=payload
        latent.update(payload=payload, output_positions=positions)
    elif task == 'shift':
        distance=1+v%3; clean=x.copy()
        positions=list(range(distance,n)); values=clean[:-distance]
        latent.update(clean=clean,distance=distance)
    elif task == 'marker':
        at=int(g.integers(0,n-2)); value=int(g.integers(3,vocab))
        x[at]=1; x[at+1]=value; x[-1]=2
        positions=[n-1]; values=[value]; latent.update(marker_position=at,value=value)
    elif task == 'parity':
        bits=g.integers(0,2,size=n-1).tolist(); selected=list(range(0,len(bits),2))
        x=[b+3 for b in bits]+[2]; positions=[n-1]
        values=[3+sum(bits[i] for i in selected)%2]
        latent.update(bits=bits,selected=selected)
    elif task == 'associative':
        count=min((n-2)//2, max(2,(vocab-3)//2)); keys=g.choice(np.arange(3,vocab),size=count,replace=False).tolist()
        # Split-specific key/value pair sets are disjoint by construction.
        s=SPLITS.index(split); choices=np.arange(3+s,vocab,3)
        vals=g.choice(choices,size=count).tolist(); query=int(g.integers(count))
        x=[t for pair in zip(keys,vals) for t in pair]+[keys[query],2]
        n=len(x); positions=[n-1]; values=[vals[query]]
        latent.update(keys=keys,values=vals,query=query)
    else:
        a,b,c=g.choice(np.arange(3,vocab),size=3,replace=False).tolist()
        x=[a,b,c,a,b,2]; n=len(x);positions=[n-1];values=[c]
        latent.update(fragment=[a,b,c])
    latent['length']=n
    observed=x.copy(); noise=rng(lab,v,role+'/noise')
    for i in range(n):
        if x[i]>=3 and noise.random()<cfg['noise']:
            observed[i]=int(noise.integers(3,vocab))
    target=[-100]*size; mask=[0]*size
    for i,t in zip(positions,values):target[i]=int(t);mask[i]=1
    row=dict(id=f'L{lab}-V{v:02d}-{split}-{index:06d}',variant=v,
             input_ids=observed+[0]*(size-n), target_ids=target,
             attention_mask=[1]*n+[0]*(size-n), loss_mask=mask,latent=latent)
    row['checksum']=sha(canonical(row)); return row

def oracle(row):
    """Generator semantic oracle; not a model solution to corrupted observations."""
    d=row['latent'];t=d['task'];n=d['length']
    if t=='copy':return dict(zip(d['output_positions'],d['payload']))
    if t=='shift':return {i:d['clean'][i-d['distance']] for i in range(d['distance'],n)}
    if t=='marker':return {n-1:d['value']}
    if t=='parity':return {n-1:3+sum(d['bits'][i] for i in d['selected'])%2}
    if t=='associative':return {n-1:d['values'][d['query']]}
    return {n-1:d['fragment'][-1]}

def validate(row,cfg):
    base={k:v for k,v in row.items() if k!='checksum'}
    if sha(canonical(base))!=row['checksum']:raise ValueError('checksum mismatch')
    x=np.array(row['input_ids']);y=np.array(row['target_ids']);a=np.array(row['attention_mask']);m=np.array(row['loss_mask'])
    if any(len(row[k])!=cfg['max_length'] for k in ('input_ids','target_ids','attention_mask','loss_mask')):raise ValueError('shape')
    if not np.all((x>=0)&(x<cfg['vocab_size'])):raise ValueError('token range')
    n=int(a.sum())
    if not np.array_equal(a,np.r_[np.ones(n,int),np.zeros(len(a)-n,int)]):raise ValueError('padding mask')
    if np.any(x[:n]==0) or np.any(x[n:]!=0):raise ValueError('padding token')
    if not np.all((m==0)|(m==1)) or not m.any() or np.any(m>a):raise ValueError('loss mask')
    if np.any(y[m==0]!=-100) or np.any((y[m==1]<0)|(y[m==1]>=cfg['vocab_size'])):raise ValueError('target range')
    actual={int(i):int(y[i]) for i in np.flatnonzero(m)}
    if actual!=oracle(row):raise ValueError('semantic oracle')

def arrays_hash(arr):
    a=np.ascontiguousarray(arr)
    return sha(canonical({'shape':list(a.shape),'dtype':str(a.dtype)})+a.tobytes())

def generate(lab,variant,destination,n=64):
    cfg=parameters(variant,lab)
    if n<1:raise ValueError('n must be positive')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    def write_json(name,obj): (out/name).write_bytes(canonical(obj)+b'\n')
    write_json('config.json',cfg)
    logical={}
    def save_npz(name,**arrays):
        np.savez_compressed(out/name,**arrays)
        logical[name]={k:arrays_hash(v) for k,v in arrays.items()}
    if lab in (1,2):
        all_ids=set()
        for split,count in [('train',n),('valid',max(2,n//4)),('test',max(2,n//4))]:
            rows=[sample(cfg,split,i) for i in range(count)]
            for row in rows:
                validate(row,cfg)
                if row['id'] in all_ids:raise ValueError('duplicate id')
                all_ids.add(row['id'])
            (out/f'{split}.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
            arr={k:np.array([r[k] for r in rows]) for k in ('input_ids','target_ids','attention_mask','loss_mask')}
            arr['ids']=np.array([r['id'] for r in rows]);save_npz(f'{split}.npz',**arr)
        if lab==1:
            g=rng(lab,variant,'attention');h=cfg['heads'];t=cfg['max_length'];d=cfg['head_dim']
            allowed=np.ones((t,t),bool)
            if cfg['mask_type']=='causal':allowed=np.tri(t,dtype=bool)
            if cfg['mask_type']=='window':allowed=np.abs(np.arange(t)[:,None]-np.arange(t)[None,:])<=2
            save_npz('attention_inputs.npz',q=g.normal(size=(h,t,d)),k=g.normal(size=(h,t,d)),v=g.normal(size=(h,t,d)),allowed=allowed)
    elif lab==3:
        g=rng(lab,variant,'control-representations');d=cfg['model_width'];x=g.normal(size=(n,d));rot=np.linalg.qr(g.normal(size=(d,d)))[0];permutation=g.permutation(n)
        save_npz('control_representations.npz',ids=np.array([f'control-{i:05d}' for i in range(n)]),
                 x=x,y=x@rot+.01*g.normal(size=x.shape),y_unpaired=(x@rot)[permutation],
                 permutation=permutation,labels=(x[:,0]>0).astype(int))
        write_json('provenance.json',{'kind':'synthetic_control','trained_model':False,
          'instruction':'For the main study replace these controls with paired activations and IDs from your lab 2 checkpoint.'})
    else:
        g=rng(lab,variant,'simulated-evaluation');records=[]
        for split in ('valid','test'):
            for i in range(n):
                label=int(g.random()<.3);z=(2*label-1)*1.3+g.normal()
                records.append({'id':f'{split}-{i:05d}','split':split,'target':label,
                  'probabilities':{k:float(1/(1+np.exp(-z*scale+shift))) for k,scale,shift in [('A',1.,0.),('B',1.3,.1),('C',.7,-.2)]},
                  'stress_probability':float(1/(1+np.exp(-(z+g.normal(0,.7))))),
                  'ablation_probability':float(1/(1+np.exp(-(z+g.normal(0,1.))))),
                  'destructive_probability':float(g.random())})
        (out/'predictions.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in records))
        write_json('resources.json',{'kind':'simulated_not_measured','configurations':[
            {'name':'A','latency_ms':4.,'memory_mib':60.}, {'name':'B','latency_ms':7.,'memory_mib':90.}, {'name':'C','latency_ms':3.,'memory_mib':40.}]})
        write_json('provenance.json',{'kind':'synthetic_control_predictions','trained_model':False,
            'instruction':'Tune thresholds on valid only. Resource numbers are simulated, not hardware benchmarks. Stress and ablation describe synthetic score controls, not interventions on a trained transformer.'})
    manifest={'schema_version':SCHEMA,'generator_version':VERSION,'lab':lab,'variant':variant,
              'table_seed':cfg['table_seed'],'seed_convention':'int(SHA256(f"{lab}|{variant}|{role}|1.0")[:16],16) % 2**63',
              'rng':'NumPy PCG64','padding_id':0,'ignore_index':-100,
              'environment':{'python':platform.python_version(),'numpy':np.__version__},
              'generator_sha256':sha(Path(__file__).read_bytes()),'logical_arrays':logical,
              'sha256':{p.name:sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json('manifest.json',manifest)
    return manifest

def verify(directory):
    root=Path(directory);m=json.loads((root/'manifest.json').read_text())
    for name,digest in m['sha256'].items():
        if sha((root/name).read_bytes())!=digest:raise ValueError('file checksum: '+name)
    for name,hashes in m['logical_arrays'].items():
        with np.load(root/name,allow_pickle=False) as a:
            for k,h in hashes.items():
                if arrays_hash(a[k])!=h:raise ValueError('logical array: '+name+'/'+k)
    if m['lab'] in (1,2):
        cfg=json.loads((root/'config.json').read_text());ids=set();pair_splits=[]
        for split in SPLITS:
            pairs=set()
            rows=[json.loads(s) for s in (root/f'{split}.jsonl').read_text().splitlines()]
            with np.load(root/f'{split}.npz',allow_pickle=False) as a:
                for index,row in enumerate(rows):
                    validate(row,cfg)
                    if row['id'] in ids:raise ValueError('split leakage')
                    ids.add(row['id'])
                    for key in ('input_ids','target_ids','attention_mask','loss_mask'):
                        if not np.array_equal(a[key][index],row[key]):raise ValueError('JSONL/NPZ mismatch')
                    if str(a['ids'][index])!=row['id']:raise ValueError('ID mismatch')
                    if cfg['task']=='associative':pairs.update(zip(row['latent']['keys'],row['latent']['values']))
            pair_splits.append(pairs)
        if any(pair_splits[i]&pair_splits[j] for i in range(3) for j in range(i)):raise ValueError('associative pair leakage')
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lab',type=int,choices=range(1,5));p.add_argument('--variant',type=int);p.add_argument('--out',type=Path);p.add_argument('--n',type=int,default=64);p.add_argument('--verify',type=Path)
    a=p.parse_args()
    if a.verify:verify(a.verify);print('Verified:',a.verify);return
    if a.lab is None or a.variant is None:p.error('--lab and --variant are required')
    cfg=parameters(a.variant,a.lab);destination=a.out or Path('runs')/f'lab{a.lab}_variant{a.variant:02d}'
    generate(a.lab,a.variant,destination,a.n);verify(destination);print('Created:',destination)

if __name__=='__main__':main()
