"""Synthetic render/export proof, adversarial integrity and CSV-byte tests."""
import contextlib,hashlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import guard_only_common as c
import finalize_guard_only as r
import verify_guard_only as v
from test_verify_refinement import make_v2,freeze

def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def record(p):return dict(path=str(p.resolve()),sha256=c.sha(p),bytes=p.stat().st_size)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def rejection_fixture(root,checkpoint):
    paths=[];baseline=dict(AP_small=.1,AP50_95=.3);op=dict(precision=.5,small_recall=.2,F2=.5)
    for kind in ('initial','selection','consensus'):
        folder=root/kind;folder.mkdir();config=dict(checkpoint_sha256=checkpoint,kind=kind)
        report=dict(status='complete',split='val',images=288,checkpoint_sha256=checkpoint,configuration_sha256=digest(config),baseline=baseline)
        if kind=='initial':report.update(tiled=dict(AP_small=.11,AP50_95=.28),eligible_by_frozen_rule=False);path=folder/'METRICS.json';link='metrics'
        else:
            candidates=[]
            for i in range(12):
                checks=dict(precision=True,small_recall=True,F2=True,AP_small=False,AP50_95=True)
                candidates.append(dict(grid_index=i,metrics=dict(AP_small=.09,AP50_95=.3),operating=dict(precision=.5,small_recall=.3,F2=.6),eligible=False,eligibility_checks=checks))
            report.update(baseline_operating=op,candidates=candidates,selected_eligible=False,selected_config=None,selected=None,selected_operating=None,selected_grid_index=None);path=folder/'SELECTION.json';link='selection'
        dump(path,report);manifest=dict(status='complete',split='val',image_count=288,test_images_or_annotations_opened=False,inputs_unchanged=True,configuration=config,configuration_sha256=digest(config),inputs=[],output_files=[record(path)])
        manifest[link]=record(path);dump(folder/'EVALUATION_MANIFEST.json',manifest);paths.append(path)
    return paths

class RawCSVTests(unittest.TestCase):
    def test_multiline_comma_crlf_preserved(self):
        source='frame_index,observation_id,value\r\n0,a,"a,b\r\nc"\r\n1,b,z\r\n'
        reader=c.RawCSV(io.StringIO(source,newline=''));self.assertEqual(reader.header,'frame_index,observation_id,value\r\n')
        one=reader.frame(0);self.assertEqual(one[0][0]['value'],'a,b\r\nc');self.assertEqual(reader.header+one[0][1]+reader.pop()[1],source)
    def test_out_of_order_rejected(self):
        reader=c.RawCSV(io.StringIO('frame_index,value\n0,x\n'))
        with self.assertRaisesRegex(ValueError,'Out-of-order'):reader.frame(1)

class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);_,self.base=make_v2(self.root)
        bm=c.read(self.base/'MANIFESTE.json');checkpoint=bm['inputs_sha256'][str(Path(bm['parameters']['litter_model']).resolve())]
        self.reports=rejection_fixture(self.root,checkpoint);self.out=self.root/'guard'
        self.argv=['finalize_guard_only.py','--baseline',str(self.base),'--out',str(self.out)]
        for p in self.reports:self.argv+=['--rejection-report',str(p)]
        with patch.object(sys,'argv',self.argv),contextlib.redirect_stdout(io.StringIO()):r.main()
    def tearDown(self):self.tmp.cleanup()
    def txt(self,i,fn):
        p=self.out/f'TXT/frame_{i:06d}.txt';d=c.read(p);fn(d);dump(p,d)
    def test_actual_renderer_full_proof_and_historical_tracking(self):
        proof=v.verify_delivery(self.out);self.assertEqual((proof['frames_checked'],proof['objects_checked'],proof['quarantined_objects_checked']),(3,1,1))
        self.assertTrue(proof['retained_csv_and_points_byte_equality']);self.assertTrue(proof['historical_ids_and_tracking_preserved'])
        old=c.read(self.base/'TXT/frame_000002.txt')['objects'][0];new=c.read(self.out/'TXT/frame_000002.txt')['quarantined_predictions'][0]
        self.assertEqual(old,{k:new[k] for k in old});self.assertFalse(c.read(self.out/'MANIFESTE.json')['small_object_gain_validated'])
    def test_contour_tamper_rejected(self):
        self.txt(0,lambda d:d['objects'][0].update(full_predicted_mask_contours=[]))
        with self.assertRaisesRegex(ValueError,'Original object'):v.verify_delivery(self.out)
    def test_historic_track_tamper_rejected(self):
        self.txt(0,lambda d:d['objects'][0].update(track_id='new-track'))
        with self.assertRaisesRegex(ValueError,'Original object'):v.verify_delivery(self.out)
    def test_missing_quarantine_rejected(self):
        self.txt(2,lambda d:d.update(quarantined_predictions=[]))
        with self.assertRaisesRegex(ValueError,'Original object'):v.verify_delivery(self.out)
    def test_quarantine_points_cannot_leak(self):
        with (self.base/'POINTS_OBJETS.csv').open(newline='') as f:
            reader=c.RawCSV(f);reader.frame(0);reader.frame(1);rows=reader.frame(2)
        with (self.out/'POINTS_OBJETS.csv').open('a',newline='') as f:f.write(rows[0][1])
        freeze(self.out)
        with self.assertRaisesRegex(ValueError,'Extra output CSV'):v.verify_delivery(self.out)
    def test_small_gain_claim_rejected(self):
        p=self.out/'MANIFESTE.json';d=c.read(p);d['small_object_gain_validated']=True;dump(p,d)
        with self.assertRaisesRegex(ValueError,'must not claim'):v.verify_delivery(self.out)
    def test_csv_format_change_even_same_numeric_values_rejected(self):
        p=self.out/'OBJETS.csv';original=p.read_bytes();p.write_bytes(original.replace(b'\r\n',b'\n'));freeze(self.out)
        with self.assertRaisesRegex(ValueError,'headers changed'):v.verify_delivery(self.out)
    def test_existing_output_is_never_overwritten(self):
        before=c.sha(self.out/'MANIFESTE.json')
        with patch.object(sys,'argv',self.argv),self.assertRaisesRegex(ValueError,'already exists'):r.main()
        self.assertEqual(before,c.sha(self.out/'MANIFESTE.json'))
    def test_forged_winner_cannot_be_relabelled_rejection(self):
        p=self.reports[-1];d=c.read(p);d['selected_eligible']=True;dump(p,d)
        bm=c.read(self.base/'MANIFESTE.json');cp=bm['inputs_sha256'][str(Path(bm['parameters']['litter_model']).resolve())]
        with self.assertRaisesRegex(ValueError,'eligible winner'):c.rejection_evidence(self.reports,cp)
    def test_frame_audit_must_match_guard(self):
        p=self.out/'AUDIT_FUSION.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[2]['quarantined_observation_ids']=[];p.write_text(''.join(json.dumps(x)+'\n' for x in rows));freeze(self.out)
        with self.assertRaisesRegex(ValueError,'Guard audit'):v.verify_delivery(self.out)

if __name__=='__main__':unittest.main()
