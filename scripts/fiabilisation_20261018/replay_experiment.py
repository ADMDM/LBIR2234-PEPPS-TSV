"""Fixed-budget three-seed adaptation pilot. Never reads the public test split."""
import argparse,csv,json,math,random,sys,time,gc,hashlib
from pathlib import Path
import cv2,numpy as np,torch
P=argparse.ArgumentParser();P.add_argument('--root',type=Path,required=True);P.add_argument('--out',type=Path,required=True)
A=P.parse_args();ROOT=A.root;OUT=A.out;sys.path.insert(0,str(ROOT))
from railtsv.common import read_manifest,write_manifest,sha256,check_disjoint,seed_all
from railtsv.data import RailDataset,letterbox,normalize
from railtsv.model import build_model,freeze_batchnorm,transfer,dice_ce
from railtsv.engine import predict_image,environment
from railtsv.common import rgb_read,mask_read
from railtsv.geometry import corridor_profile

SEEDS=[19,42,73];STEPS=780;PUBLIC_EVERY=3;HEIGHT=480;WIDTH=640
EXP=ROOT/'runs/experience_egopath_20261016_141426';INIT=EXP/'preapprentissage_egopath/best.pt'
OUT.mkdir(parents=True,exist_ok=False);(OUT/'manifests').mkdir()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def order(rows):return sorted(rows,key=lambda r:hashlib.sha256(Path(r['image']).name.encode()).hexdigest())
tsv=read_manifest(EXP/'dataset/train.csv');val=read_manifest(EXP/'dataset/val.csv')
public_train=read_manifest(ROOT/'data/prepared/railsem_egopath_20261014/train.csv')
public_val=read_manifest(ROOT/'data/prepared/railsem_egopath_20261014/val.csv')
pub=order(public_train)[:260];pubval=order(public_val)[:80]
for tr in [tsv,pub]:
    for va in [val,pubval]:check_disjoint(tr,va)
for name,data in [('tsv_train',tsv),('tsv_dev',val),('public_replay_train',pub),('public_dev_80',pubval)]:write_manifest(OUT/'manifests'/f'{name}.csv',data)
protocol=dict(design='Paired three-seed adaptation pilot, same public initialization and fixed 780 optimizer updates per arm.',
 seeds=SEEDS,arms=['tsv_only','tsv_public_replay'],optimizer='AdamW lr=1e-4 weight_decay=1e-4; cosine over 780 updates; gradient clip=1',
 steps=STEPS,batch_size=1,network_size=[WIDTH,HEIGHT],initialization_sha256=sha256(INIT),
 tsv_only_exposures=780,replay_tsv_exposures=520,replay_public_exposures=260,
 public_boundary_ignore_radius_input_px=3,public_selection='260 hash-ordered images from public TRAIN only; same pool for all seeds',
 evaluation='12 existing TSV development images and 80 fixed hash-ordered public VALIDATION images; public TEST is not read.',
 checkpoint_selection='Final update only, no early stopping or best-checkpoint selection.',
 semantic_warning='Public path and TSV inner-edge corridor are different supervision targets. Public boundary band is ignored, but this does not prove target equivalence.',
 interpretation='Equal update budget, different TSV exposure counts. Tests the replay strategy, not the historical causal effect of ego-path versus extra pretraining.',
 limits='One TSV development capture; conditional variability of adaptation only, public pretraining remains fixed; no independent final test or automatic promotion.',
 training=True,environment=environment(),reference='https://arxiv.org/abs/2004.07211')
dump(OUT/'PROTOCOLE_AVANT_ENTRAINEMENT.json',protocol)
torch.set_num_threads(4);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def synchronize():
    if dev.type=='mps':torch.mps.synchronize()

def evaluate(model,rows,label):
    output=[];model.eval()
    for row in rows:
        im=rgb_read(row['image']);gt=mask_read(row['mask'],[0,1,255]);pred,probs=predict_image(model,im,dev,HEIGHT,WIDTH)
        valid=gt!=255;truth=(gt==1)&valid;pos=(pred==1)&valid
        tp=np.count_nonzero(truth&pos);fp=np.count_nonzero(pos&~truth);fn=np.count_nonzero(truth&~pos)
        profile={p['y']:p for seg in corridor_profile(pred,probs)['segments'] for p in seg}
        errors=[];ref_sections=0
        for y in range(0,gt.shape[0],8):
            xs=np.flatnonzero(gt[y]==1)
            if len(xs)<2 or np.any(np.diff(xs)>1):continue
            # Require known adjacent background to avoid a reference clipped by ignore.
            if xs[0]==0 or xs[-1]==gt.shape[1]-1 or gt[y,xs[0]-1]!=0 or gt[y,xs[-1]+1]!=0:continue
            ref_sections+=1
            if y in profile:errors.append(profile[y]['center_x']-(xs[0]+xs[-1])/2)
        output.append(dict(domain=label,image=Path(row['image']).stem,sequence=row['sequence'],width=gt.shape[1],height=gt.shape[0],
            dice=(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
            reference_sections=ref_sections,observed_sections=len(errors),mae_px=float(np.mean(np.abs(errors))) if errors else None,
            mae_pct_width=float(100*np.mean(np.abs(errors))/gt.shape[1]) if errors else None))
    return output

results=[];start_all=time.time()
for seed in SEEDS:
    for arm in ['tsv_only','tsv_public_replay']:
        dest=OUT/f'{arm}_seed_{seed}';dest.mkdir();seed_all(seed)
        tsv_ds=RailDataset(OUT/'manifests/tsv_train.csv',2,HEIGHT,WIDTH,True,seed)
        pub_ds=RailDataset(OUT/'manifests/public_replay_train.csv',2,HEIGHT,WIDTH,True,seed+1000)
        model=build_model(2);transfer(model,INIT);model.to(dev).train();freeze_batchnorm(model)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS)
        rng=np.random.default_rng(seed);tsv_indices=[]
        while len(tsv_indices)<780:tsv_indices.extend(rng.permutation(len(tsv_ds)).tolist())
        pub_indices=np.random.default_rng(seed+2000).permutation(len(pub_ds)).tolist()
        ti=pi=0;history=[];begin=time.perf_counter();losses=[]
        dump(OUT/'ETAT.json',dict(status='training',arm=arm,seed=seed,step=0,total_models=6))
        for step in range(STEPS):
            is_public=arm=='tsv_public_replay' and (step+1)%PUBLIC_EVERY==0
            if is_public:
                im,target=pub_ds[pub_indices[pi]];pi+=1
                a=target.numpy().astype(np.uint8);fg=(a==1).astype(np.uint8)
                boundary=cv2.morphologyEx(fg,cv2.MORPH_GRADIENT,np.ones((7,7),np.uint8))>0
                a[boundary]=255;target=torch.from_numpy(a.astype(np.int64))
            else:im,target=tsv_ds[tsv_indices[ti]];ti+=1
            assert (target!=255).any()
            model.train();freeze_batchnorm(model);opt.zero_grad(set_to_none=True)
            loss=dice_ce(model(im[None].to(dev)),target[None].to(dev));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();schedule.step()
            losses.append(float(loss.detach().cpu()))
            if (step+1)%130==0:
                rec=dict(step=step+1,loss_mean=float(np.mean(losses[-130:])),tsv_exposures=ti,public_exposures=pi,seconds=time.perf_counter()-begin)
                history.append(rec);dump(dest/'history.json',history)
                dump(OUT/'ETAT.json',dict(status='training',arm=arm,seed=seed,step=step+1,total_models=6))
                print(f'{arm} seed={seed} step={step+1}/{STEPS} loss={rec["loss_mean"]:.4f} elapsed={rec["seconds"]:.0f}s',flush=True)
        synchronize()
        torch.save(dict(format='railtsv-deeplab-v1',architecture='DeepLabV3Plus',encoder='resnet50',
            state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},task='corridor',classes=['background','couloir'],
            height=HEIGHT,width=WIDTH,preprocessing='imagenet-letterbox-v1',ignore_index=255,epoch=6,steps=STEPS,
            synthetic_demo=False,training_only_demo=False,seed=seed,development_pilot=True,variant=arm),dest/'last.pt')
        measured=evaluate(model,val,'tsv_dev_12')+evaluate(model,pubval,'public_dev_80')
        with (dest/'EVALUATION_PAR_IMAGE.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(measured[0]));w.writeheader();w.writerows(measured)
        for domain in ['tsv_dev_12','public_dev_80']:
            part=[r for r in measured if r['domain']==domain]
            results.append(dict(arm=arm,seed=seed,domain=domain,images=len(part),steps=STEPS,tsv_exposures=ti,public_exposures=pi,
                dice_macro=float(np.mean([r['dice'] for r in part])),iou_macro=float(np.mean([r['iou'] for r in part])),
                axis_coverage_pct=100*sum(r['observed_sections'] for r in part)/sum(r['reference_sections'] for r in part),
                image_mean_mae_pct_width=float(np.mean([r['mae_pct_width'] for r in part if r['mae_pct_width'] is not None])),
                seconds=time.perf_counter()-begin))
        dump(OUT/'RESULTATS_INTERMEDIAIRES.json',results)
        del model,opt,schedule,tsv_ds,pub_ds;gc.collect()
        if dev.type=='mps':torch.mps.empty_cache()
with (OUT/'COMPARAISON_TROIS_GRAINES.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
dump(OUT/'TERMINE.json',dict(complete=True,models=6,seeds=SEEDS,elapsed_s=time.time()-start_all,final_test_used=False,production_model_replaced=False))
dump(OUT/'ETAT.json',dict(status='completed',models=6));print('EXPERIMENT_COMPLETE',flush=True)
