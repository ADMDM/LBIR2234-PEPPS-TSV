import json,tempfile,unittest
from pathlib import Path
import numpy as np
from corridor_geometry import analyze_object_mask
from refine_video import baseline_objects,decorate,serial_record,merge_public_masks,summarize
from lire_contours import load_frame,get_objects

class RefineTests(unittest.TestCase):
    def obj(self,source='litter',raw=0,candidate='t0000'):
        mask=np.zeros((10,12),bool);mask[3:7,4:8]=True
        return dict(mask=mask,mask_bbox=[4,3,8,7],model_source=source,class_id=0,
            class_name='dechet' if source=='litter' else 'person',confidence_score=.8,
            raw_detection_index=raw,model_bbox_original_px=[4.,3.,8.,7.],inference_crop_xyxy=None,
            touches_crop_border=False,candidate_id=candidate,view_id='tile_0000',view_kind='tile',tile_index=0)
    def test_decoration_and_baseline_reconstruction(self):
        obj=self.obj();corridor=np.ones((10,12),bool);ar=analyze_object_mask(obj['mask'],corridor,corridor)
        obj=decorate(obj,corridor,ar,'V:v2:f000000:o003');obj.update(track_id=None,track_confirmed=False,observed_hits=0)
        self.assertEqual(obj['display_number'],4)
        data=serial_record(obj);json.dumps(data,allow_nan=False)
        restored=baseline_objects({'objects':[data]},12,10)
        np.testing.assert_array_equal(restored[0]['mask'],obj['mask'])
        self.assertEqual(restored[0]['baseline_observation_id'],data['observation_id'])
        data['full_mask_sha256']='0'*64
        with self.assertRaises(ValueError):baseline_objects({'objects':[data]},12,10)
    def test_public_duplicate_trace_uses_unique_candidates(self):
        pub=self.obj('public',0,'b0002');one=self.obj('litter',0,'t0000');two=self.obj('litter',0,'t0001')
        two['mask']=np.zeros((10,12),bool);two['mask'][0:2,0:2]=True;two['mask_bbox']=[0,0,2,2]
        kept,audit=merge_public_masks([pub],[one,two])
        self.assertEqual([o['candidate_id'] for o in kept],['b0002','t0001'])
        self.assertEqual(audit[0]['discarded_candidate_id'],'t0000')
        self.assertEqual(audit[0]['kept_candidate_id'],'b0002')
    def test_quarantine_decoration_has_separate_identifier(self):
        obj=self.obj();m=np.ones((10,12),bool)
        x=decorate(obj,m,analyze_object_mask(obj['mask'],m,m),'V:v2:f000000:q002')
        self.assertEqual(x['display_number'],3)
        self.assertNotIn('mask',serial_record(x))

    def test_quarantined_train_cannot_suppress_litter(self):
        pub=self.obj('public',0,'b0001');pub.update(class_name='train',confidence_score=.3)
        pub['mask']=np.ones((10,12),bool);pub['mask_bbox']=[0,0,12,10]
        lit=self.obj('litter',0,'t0001');lit['mask']=pub['mask'].copy();lit['mask_bbox']=[0,0,12,10]
        kept,audit=merge_public_masks([pub],[lit],np.ones((10,12),bool))
        self.assertEqual(len(kept),2);self.assertEqual(audit,[])

if __name__=='__main__':unittest.main()
