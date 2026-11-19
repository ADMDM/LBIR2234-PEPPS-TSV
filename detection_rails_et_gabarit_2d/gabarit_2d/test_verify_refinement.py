"""Adversarial CPU fixtures for V2 export verification; no ML model is loaded."""
import copy,csv,json,shutil,tempfile,unittest
from pathlib import Path
import numpy as np
import infer_gabarit_2d as e
import verify_gabarit_2d as v1
import verify_refinement as v2
from corridor_geometry import corridor_from_rows,analyze_object_mask,extract_contours
from refine_video import summarize,decorate,serial_record,EXTRA_FRAME,REVIEW_FIELDS
from small_object_refinement import train_guard,tile_windows
from test_verify_gabarit_2d import make_delivery,dump,table


def read_rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def freeze(run):
    m=json.loads((run/'MANIFESTE.json').read_text())
    m['output_sha256']={n:v2.sha(run/n) for n in m['output_sha256']}
    dump(run/'MANIFESTE.json',m)

def replace_baseline_masks(run):
    """Make one tall low-score synthetic train in a valid V1 baseline."""
    bm=json.loads((run/'MANIFESTE.json').read_text())
    model=next(Path(p) for p in bm['inputs_sha256'] if p.endswith('.pt'))
    bm['parameters'].update(litter_model=str(model),public_model=str(model),litter_conf=.15)
    bm['model_names']={'public':{'0':'objet_synthetique','1':'train'},'litter':{'0':'dechet'}}
    frames=read_rows(run/'FRAMES.csv');axes=read_rows(run/'RAILS.csv')
    objects=[];audits=[];point_count=0
    with (run/'POINTS_OBJETS.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=e.POINT_FIELDS);writer.writeheader()
        for index in range(3):
            p=run/f'TXT/frame_{index:06d}.txt';d=json.loads(p.read_text());base={k:d[k] for k in e.COMMON}
            if index==2:
                rr=[r for r in axes if int(r['frame_index'])==index];c,s,_=corridor_from_rows(rr,32,24)
                mask=np.zeros((24,32),bool);mask[2:24,12:19]=True
                o=d['objects'][0]
                o.update(model_source='public',class_id=1,class_name='train',confidence_score=.3,
                         mask_bbox=[12,2,19,24],model_bbox_original_px=[12.,2.,19.,24.],
                         track_id='0:public:1:1',track_confirmed=False,observed_hits=1,
                         analysis=analyze_object_mask(mask,c,s),full_mask_sha256=e.mask_sha(mask),
                         intersection_mask_sha256=e.mask_sha(mask&c),full_predicted_mask_contours=extract_contours(mask),
                         corridor_intersection_contours=extract_contours(mask&c))
                frames[index]['confirmed_overlap_candidates']=0
            dump(p,d)
            for o in d['objects']:
                objects.append(summarize(base,o,str(p.relative_to(run))))
                point_count+=e.write_points(writer,base,o,'full_predicted_mask',o['full_predicted_mask_contours'])
                point_count+=e.write_points(writer,base,o,'corridor_intersection',o['corridor_intersection_contours'])
            audits.append(dict(base,suppressed=[],empty_masks=0,merged_predictions=[{**{k:o[k] for k in ('model_source','raw_detection_index','class_name','confidence_score','mask_bbox')},**{k:o['analysis'][k] for k in ('relation_2d','overlap_pixel_count','support_status')}} for o in d['objects']]))
    table(run/'OBJETS.csv',e.OBJECT_FIELDS,objects);table(run/'FRAMES.csv',e.FRAME_FIELDS,frames)
    (run/'AUDIT_FUSION.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in audits))
    bm['point_rows']=point_count;bm['output_sha256']={n:v1.sha(run/n) for n in bm['output_sha256']};dump(run/'MANIFESTE.json',bm)
    v1.verify_delivery(run)
    return model


def make_v2(root):
    (root/'baseline_fixture').mkdir();baseline=make_delivery(root/'baseline_fixture');model=replace_baseline_masks(baseline)
    bm=json.loads((baseline/'MANIFESTE.json').read_text());bv=json.loads((baseline/'VERIFICATION.json').read_text())
    out=root/'v2';out.mkdir();(out/'TXT').mkdir()
    for n in ('VIDEO_GABARIT_2D.mp4','RAILS.csv'):shutil.copy2(baseline/n,out/n)
    axes=read_rows(out/'RAILS.csv');old_frames=read_rows(baseline/'FRAMES.csv')
    objects=[];reviews=[];frames=[];audits=[];pc=0
    with (out/'POINTS_OBJETS.csv').open('w',newline='') as handle:
        pw=csv.DictWriter(handle,fieldnames=e.POINT_FIELDS);pw.writeheader()
        for index in range(3):
            old=json.loads((baseline/f'TXT/frame_{index:06d}.txt').read_text());base={k:old[k] for k in e.COMMON}
            rr=[r for r in axes if int(r['frame_index'])==index];c,s,ri=corridor_from_rows(rr,32,24)
            windows=tile_windows(e.crop_for_corridor(c,.15),32,24,16,.25)
            views=[dict(view_id=f'tile_{i:04d}',view_kind='tile',tile_index=i,crop_xyxy=w) for i,w in enumerate(windows)]
            group=[];qgroup=[]
            for j,oldobj in enumerate(old['objects']):
                o=copy.deepcopy(oldobj);o.update(candidate_id=f'b{j:04d}',inference_origin='baseline',baseline_observation_id=oldobj['observation_id'],view_id='baseline',view_kind='global',tile_index=None)
                mask=v1.decode_contours(o['full_predicted_mask_contours'],32,24)
                guard=train_guard(dict(o,mask=mask),c)
                if guard['quarantine']:
                    o.update(observation_id=f"{base['video_id']}:v2:f{index:06d}:q000",display_number=1,track_id=None,track_confirmed=False,observed_hits=0,review_reason=guard['reason'],review_geometry=guard);qgroup.append(o)
                else:
                    o.update(observation_id=f"{base['video_id']}:v2:f{index:06d}:o000",display_number=1);group.append(o)
            if index==1:
                mask=np.zeros((24,32),bool);mask[10:12,13:15]=True
                o=dict(model_source='litter',class_id=0,class_name='dechet',confidence_score=.7,raw_detection_index=0,
                       mask=mask,mask_bbox=[13,10,15,12],model_bbox_original_px=[13.,10.,15.,12.],inference_crop_xyxy=windows[0],touches_crop_border=False,
                       candidate_id='t0000',inference_origin='tile',baseline_observation_id=None,view_id='tile_0000',view_kind='tile',tile_index=0,tile_detection_index=0,
                       track_id=None,track_confirmed=False,observed_hits=0)
                o=serial_record(decorate(o,c,analyze_object_mask(mask,c,s),f"{base['video_id']}:v2:f{index:06d}:o000"));group.append(o)
            rel=f'TXT/frame_{index:06d}.txt'
            dump(out/rel,dict(base,schema_version=2,processing_status='ok',coordinate_convention='original pixel centres, x right, y down',rail_diagnostic_status=old['rail_diagnostic_status'],rail_support=ri,objects=group,quarantined_predictions=qgroup))
            for o in group:
                objects.append(summarize(base,o,rel));pc+=e.write_points(pw,base,o,'full_predicted_mask',o['full_predicted_mask_contours']);pc+=e.write_points(pw,base,o,'corridor_intersection',o['corridor_intersection_contours'])
            for o in qgroup:reviews.append(dict(summarize(base,o,rel),review_reason=o['review_reason']))
            raw=[]
            for o in group+qgroup:
                x0,y0,x1,y1=o['mask_bbox'];raw.append({**{k:o[k] for k in ('candidate_id','inference_origin','model_source','raw_detection_index','class_name','confidence_score','mask_bbox')},'bbox_overlap_pixel_count':int(c[y0:y1,x0:x1].sum()),**{k:o['analysis'][k] for k in ('relation_2d','overlap_pixel_count','support_status')},'review_reason':o.get('review_reason')})
            audits.append(dict(base,suppressed=[],merged_predictions=raw,empty_masks=0,tile_views=views))
            frames.append(dict(base,processing_status='ok',rail_diagnostic_status=old['rail_diagnostic_status'],rail_supported_rows=ri['supported_rows'],rail_corridor_pixel_count=ri['corridor_pixel_count'],raw_public_instances=len(old['objects']),raw_litter_instances=int(index==1),merged_instances=len(raw),suppressed_duplicate_instances=0,empty_mask_instances=0,overlap_candidates=len(group),confirmed_overlap_candidates=sum(o['track_confirmed'] for o in group),indeterminate_instances=0,txt_file=rel,baseline_candidates=len(old['objects']),tile_instances=int(index==1),quarantined_candidates=len(qgroup),tile_views=len(views)))
    table(out/'OBJETS.csv',e.OBJECT_FIELDS,objects);table(out/'A_VERIFIER.csv',REVIEW_FIELDS,reviews);table(out/'FRAMES.csv',e.FRAME_FIELDS+EXTRA_FRAME,frames)
    (out/'AUDIT_FUSION.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in audits))
    raildir=Path(bm['parameters']['rail_delivery'])
    frozen=[Path(bm['source']),model]+[baseline/n for n in ('MANIFESTE.json','VERIFICATION.json','RAILS.csv','FRAMES.csv','OBJETS.csv')]+[raildir/n for n in ('SOURCE_ET_PARAMETRES.json','FRAMES.csv','AXE_ET_LIMITES.csv')]
    m=dict(status='complete',schema_version=2,video_id=bm['video_id'],source=bm['source'],source_sha256=bm['source_sha256'],frames=3,first_time_s=bm['first_time_s'],last_time_s=bm['last_time_s'],rails_recomputed=False,partial_source=False,rail_csv_byte_identical=True,parameters=dict(baseline=str(baseline),rail_delivery=str(raildir),litter_model=str(model),max_frames=0,tile_size=16,tile_overlap=.25,conf=.15),model_names=bm['model_names'],inputs_sha256={str(p.resolve()):v2.sha(p) for p in frozen},baseline_manifest_sha256=v2.sha(baseline/'MANIFESTE.json'),baseline_txt_path_and_content_aggregate_sha256=bv['txt_path_and_content_aggregate_sha256'],overlap_observations=len(objects),quarantined_observations=len(reviews),point_rows=pc,output_sha256={n:v2.sha(out/n) for n in v2.OUTPUTS})
    dump(out/'MANIFESTE.json',m);return out,baseline


class RefinementProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.out,self.base=make_v2(self.root)
    def tearDown(self):self.temp.cleanup()
    def edit_txt(self,index,fn):
        p=self.out/f'TXT/frame_{index:06d}.txt';d=json.loads(p.read_text());fn(d);dump(p,d)
    def test_valid_complete_delivery_with_tile_and_quarantine(self):
        r=v2.verify_delivery(self.out);self.assertEqual((r['frames_checked'],r['objects_checked'],r['quarantined_objects_checked']), (3,2,1));self.assertTrue(r['all_quarantined_masks_reconstructed'])
    def test_quarantine_must_match_guard(self):
        self.edit_txt(2,lambda d:d['quarantined_predictions'][0].update(review_reason='manual_hide'))
        with self.assertRaisesRegex(ValueError,'quarantine reason'):v2.verify_delivery(self.out)
    def test_geometry_audit_cannot_be_forged(self):
        self.edit_txt(2,lambda d:d['quarantined_predictions'][0]['review_geometry'].update(corridor_coverage=.1))
        with self.assertRaisesRegex(ValueError,'review geometry'):v2.verify_delivery(self.out)
    def test_tile_coordinates_must_match_recipe(self):
        self.edit_txt(1,lambda d:d['objects'][0].update(inference_crop_xyxy=[1,0,16,16]))
        with self.assertRaisesRegex(ValueError,'crop differs'):v2.verify_delivery(self.out)
    def test_baseline_changed_after_proof_rejected(self):
        p=self.base/'TXT/frame_000000.txt';d=json.loads(p.read_text());d['objects'][0]['confidence_score']=.9;dump(p,d)
        with self.assertRaisesRegex(ValueError,'Baseline TXT contents changed'):v2.verify_delivery(self.out)
    def test_accepted_mask_cannot_be_dropped_silently(self):
        self.edit_txt(1,lambda d:d.update(objects=[]))
        rows=read_rows(self.out/'OBJETS.csv');table(self.out/'OBJETS.csv',e.OBJECT_FIELDS,[r for r in rows if r['frame_index']!='1'])
        rows=read_rows(self.out/'POINTS_OBJETS.csv');table(self.out/'POINTS_OBJETS.csv',e.POINT_FIELDS,[r for r in rows if r['frame_index']!='1'])
        rows=read_rows(self.out/'FRAMES.csv');rows[1]['overlap_candidates']=0;table(self.out/'FRAMES.csv',e.FRAME_FIELDS+EXTRA_FRAME,rows)
        freeze(self.out)
        with self.assertRaisesRegex(ValueError,'overlap was dropped'):v2.verify_delivery(self.out)
    def test_quarantine_must_not_leak_into_main_points(self):
        p=self.out/'POINTS_OBJETS.csv'
        with p.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=e.POINT_FIELDS);d=json.loads((self.out/'TXT/frame_000002.txt').read_text());o=d['quarantined_predictions'][0];base={k:d[k] for k in e.COMMON};e.write_points(w,base,o,'full_predicted_mask',o['full_predicted_mask_contours'])
        freeze(self.out)
        with self.assertRaisesRegex(ValueError,'quarantine leaked'):v2.verify_delivery(self.out)
    def test_crop_border_flag_recomputed(self):
        self.edit_txt(1,lambda d:d['objects'][0].update(touches_crop_border=True))
        with self.assertRaisesRegex(ValueError,'Crop-border flag'):v2.verify_delivery(self.out)
    def test_wrong_tile_grid_even_with_updated_audit_hash_rejected(self):
        p=self.out/'AUDIT_FUSION.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[1]['tile_views'][0]['crop_xyxy'][0]=1;p.write_text(''.join(json.dumps(x)+'\n' for x in rows));freeze(self.out)
        with self.assertRaisesRegex(ValueError,'Tile grid differs'):v2.verify_delivery(self.out)
    def test_original_observation_candidate_id_cannot_be_forged(self):
        self.edit_txt(0,lambda d:d['objects'][0].update(candidate_id='b7777'))
        with self.assertRaisesRegex(ValueError,'Baseline candidate ID'):v2.verify_delivery(self.out)
    def test_quarantined_train_must_not_suppress_litter(self):
        p=self.out/'AUDIT_FUSION.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
        rows[2]['suppressed']=[dict(rule='public_mask_precedence_for_cross_model_duplicate',discarded_candidate_id='t0000',kept_candidate_id='b0000',discarded_view_id='tile_0000',kept_view_id='baseline',mask_iou=.7)]
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        fr=read_rows(self.out/'FRAMES.csv');fr[2].update(tile_instances=1,raw_litter_instances=1,suppressed_duplicate_instances=1);table(self.out/'FRAMES.csv',e.FRAME_FIELDS+EXTRA_FRAME,fr);freeze(self.out)
        with self.assertRaisesRegex(ValueError,'Quarantined train must not suppress litter'):v2.verify_delivery(self.out)
    def test_portable_without_original_rail_export_directory(self):
        m=json.loads((self.out/'MANIFESTE.json').read_text());old=Path(m['parameters']['rail_delivery'])
        for n in ('SOURCE_ET_PARAMETRES.json','FRAMES.csv','AXE_ET_LIMITES.csv'):
            p=old/n;m['inputs_sha256'].pop(str(p.resolve()));p.unlink()
        dump(self.out/'MANIFESTE.json',m)
        self.assertEqual(v2.verify_delivery(self.out)['status'],'passed')
    def test_new_model_checkpoint_not_silently_allowed(self):
        m=json.loads((self.out/'MANIFESTE.json').read_text());p=Path(m['parameters']['litter_model']);p.write_bytes(b'changed')
        m['inputs_sha256'][str(p.resolve())]=v2.sha(p);dump(self.out/'MANIFESTE.json',m)
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):v2.verify_delivery(self.out)

class GuardBoundaryTests(unittest.TestCase):
    def test_score_half_not_quarantined_and_compact_train_not_quarantined(self):
        c=np.zeros((100,100),bool);c[10:100,40:60]=1;m=c.copy()
        self.assertTrue(v2.independent_train_review({'class_name':'train','confidence_score':.49},m,c)[0])
        self.assertFalse(v2.independent_train_review({'class_name':'train','confidence_score':.5},m,c)[0])
        self.assertFalse(v2.independent_train_review({'class_name':'person','confidence_score':.2},m,c)[0])
        short=np.zeros_like(c);short[50:100,40:60]=1
        self.assertFalse(v2.independent_train_review({'class_name':'train','confidence_score':.2},short,c)[0])
    def test_grid_covers_crop_and_stays_in_image(self):
        c=np.zeros((77,123),bool);c[4:72,47:80]=1
        a=v2.expected_tile_views(c,32,.25);m=np.zeros_like(c)
        for v in a:
            x0,y0,x1,y1=v['crop_xyxy'];m[y0:y1,x0:x1]=1
        self.assertTrue(m[c].all());self.assertEqual([v['tile_index'] for v in a],list(range(len(a))))

if __name__=='__main__':unittest.main()
