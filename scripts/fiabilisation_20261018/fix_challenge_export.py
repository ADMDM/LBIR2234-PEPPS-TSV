from pathlib import Path
import shutil,sys
root=Path(sys.argv[1]);backup=root/'runs/fiabilisation_20261018/code_avant_correction'
p=root/'scripts/export_reviewed_challenge.py';shutil.copy2(p,backup/p.name)
s=p.read_text().replace('from railtsv.common import read_manifest,write_manifest,check_disjoint,sha256','from railtsv.common import read_manifest,write_manifest,check_disjoint,sha256\nfrom railtsv.validation_protocol import final_split_guard')
s=s.replace('def export(batch,out):','def export(batch,out,final_registry=None):')
s=s.replace("    out.mkdir(parents=True);(out/'masks').mkdir();(out/'annotations_figees').mkdir()",'''    # Historical control frames have already been inspected and used to tune
    # crop/filter parameters. They remain development data by default.
    controls=[r for r,m in accepted if r['role']=='controle_nouvelle_capture']
    final_verification=None
    if final_registry:
        registry=json.loads(Path(final_registry).read_text())
        final_verification=final_split_guard(registry,sorted({r['sequence'] for r in controls}))
        known={r['capture_id']:r for r in registry}
        for r in controls:
            if sha256(ROOT/r['source'])!=known[r['sequence']]['source_sha256']:
                raise ValueError('Source vidéo différente du registre final')
    else:
        rows.pop('test');rows['challenge_dev']=[]
    out.mkdir(parents=True);(out/'masks').mkdir();(out/'annotations_figees').mkdir()''')
s=s.replace("split='train' if r['role']=='renfort_a_annoter' else 'test'","split='train' if r['role']=='renfort_a_annoter' else ('test' if final_registry else 'challenge_dev')")
s=s.replace("for a,b in [('train','val'),('train','test'),('val','test')]:check_disjoint(rows[a],rows[b])","control_split='test' if final_registry else 'challenge_dev'\n    for a,b in [('train','val'),('train',control_split),('val',control_split)]:check_disjoint(rows[a],rows[b])")
s=s.replace("test='New capture reserved; do not select hyperparameters on this set',","control_split=control_split,final_verification=final_verification,\n        control_scope='Final only with verified frozen capture registry; otherwise development challenge, not independent test',")
s=s.replace("p.add_argument('--batch',required=True);p.add_argument('--out',required=True);a=p.parse_args();export(a.batch,a.out)","p.add_argument('--batch',required=True);p.add_argument('--out',required=True);p.add_argument('--final-registry',help='Explicit verified capture registry; omission exports challenge_dev, never test');a=p.parse_args();export(a.batch,a.out,a.final_registry)")
p.write_text(s)
p=root/'scripts/prepare_challenge_review.py';shutil.copy2(p,backup/p.name);s=p.read_text()
s=s.replace('# Entire new capture reserved from training; labels stay blank for review.','# Previously inspected capture: development challenge, NOT an independent final test.')
s=s.replace("train_candidates=30,new_capture_control_candidates=20,all_reviewed=False,training_performed=False,","train_candidates=30,development_control_candidates=20,all_reviewed=False,training_performed=False,\n        control_scope='development_exposed',predictions_seen_before_freeze=True,")
s=s.replace("note='No accuracy without human references. Same-day African clips grouped with existing photos; not independent test.'","note='No accuracy without human references. Same-day clips and TSV_record are development data. Export defaults to challenge_dev; final designation requires verified source registry.'")
p.write_text(s)
print('Historical control export now defaults to challenge_dev; final registry required for test.csv.')
