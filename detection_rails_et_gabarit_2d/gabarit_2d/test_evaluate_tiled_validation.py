import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import evaluate_tiled_validation as ev


def object_mask(mask,score=.8,view='global',edge=False):
    yy,xx=np.nonzero(mask)
    return {'mask':mask,'mask_bbox':[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],
            'confidence_score':score,'class_name':'dechet','class_id':0,'model_source':'litter',
            'raw_detection_index':0,'view_id':view,'view_kind':'global' if view=='global' else 'tile','touches_crop_border':edge}


def example_gt():
    original={'info':{},'images':[{'id':1,'width':100,'height':100,'file_name':'a.jpg'}],
              'categories':[{'id':2,'name':'paper'}], 'annotations':[{'id':7,'image_id':1,'category_id':2,'iscrowd':0,'area':200,
                'bbox':[10,10,60,60],'segmentation':[[10,10,30,10,30,30,10,30],[70,70,90,70,90,90,70,90]]}]}
    rows=[{'image_id':1,'prepared_width':50,'prepared_height':50,'scale_x':.5,'scale_y':.5,'prepared_image':'test.jpg'}]
    return original,rows


class GeometryAndMetrics(unittest.TestCase):
    def test_scale_preserves_components_and_source(self):
        original,rows=example_gt();saved=copy.deepcopy(original);gt,data=ev.prepared_ground_truth(original,rows)
        self.assertEqual(original,saved);self.assertEqual(len(data['annotations'][0]['segmentation']),2)
        mask=gt.annToMask(data['annotations'][0]);self.assertEqual(mask.shape,(50,50));self.assertTrue(mask[7,7]);self.assertTrue(mask[37,37]);self.assertFalse(mask[25,25]);self.assertEqual(data['annotations'][0]['area'],200)
    def test_bad_polygon_rejected(self):
        original,rows=example_gt();original['annotations'][0]['segmentation']=[[0,0,1]]
        with self.assertRaises(ValueError):ev.prepared_ground_truth(original,rows)
    def test_crowd_not_silently_counted_as_fn(self):
        original,rows=example_gt();original['annotations'][0]['iscrowd']=1
        with self.assertRaises(ValueError):ev.prepared_ground_truth(original,rows)
    def test_exact_operating_pr_with_duplicate_fp(self):
        original,rows=example_gt();gt,data=ev.prepared_ground_truth(original,rows);rle=gt.annToRLE(data['annotations'][0])
        pred=[{'image_id':1,'category_id':1,'score':.9,'segmentation':rle},{'image_id':1,'category_id':1,'score':.8,'segmentation':rle}]
        m=ev.operating_pr(gt,pred);self.assertEqual((m['true_positive'],m['false_positive'],m['false_negative']),(1,1,0));self.assertEqual(m['precision'],.5);self.assertEqual(m['recall'],1)
    def test_empty_operating_predictions(self):
        original,rows=example_gt();gt,_=ev.prepared_ground_truth(original,rows);m=ev.operating_pr(gt,[])
        self.assertEqual((m['true_positive'],m['false_positive'],m['false_negative']),(0,0,1));self.assertEqual(m['F2'],0)
    def test_prefilter_before_dedup_keeps_valid_operating_mask(self):
        mask=np.zeros((20,20),bool);mask[2:12,2:12]=True
        global_obj=object_mask(mask,.10);tile_obj=object_mask(mask,.60,'tile_0000',True)
        all_scores,_=ev.merge_for_threshold([global_obj],[tile_obj],.001);deployed,_=ev.merge_for_threshold([global_obj],[tile_obj],.15)
        self.assertEqual(all_scores[0]['confidence_score'],.1);self.assertEqual(deployed[0]['confidence_score'],.6)
    def test_holes_and_components_cache_roundtrip(self):
        m=np.zeros((20,30),bool);m[2:10,2:10]=True;m[4:8,4:8]=False;m[15,25]=True
        obj=object_mask(m,.7,'tile_0001');out=ev.deserialize_object(ev.serialize_object(obj),{'prepared_width':30,'prepared_height':20})
        np.testing.assert_array_equal(m,out['mask'])
    def test_cached_global_bbox_from_mask_not_detection_box(self):
        mask=np.zeros((10,12),bool);mask[3:7,4:6]=True
        objs,empty=ev.baseline_objects([{'score':.6,'bbox':[0,0,12,10],'segmentation':ev.encode_mask(mask)}],{'prepared_width':12,'prepared_height':10})
        self.assertEqual(objs[0]['mask_bbox'],[4,3,6,7]);self.assertEqual(empty,0)
    def test_empty_global_counted_not_fake_box_mask(self):
        objs,empty=ev.baseline_objects([{'score':.6,'bbox':[0,0,12,10],'segmentation':ev.encode_mask(np.zeros((10,12),bool))}],{'prepared_width':12,'prepared_height':10})
        self.assertEqual(objs,[]);self.assertEqual(empty,1)
    def test_wrong_rle_grid_rejected(self):
        with self.assertRaises(ValueError):ev.decode_mask(ev.encode_mask(np.zeros((10,12),bool)),12,11)
    def test_max_100_after_fusion(self):
        mask=np.ones((2,2),bool);objs=[dict(object_mask(mask,.2+i/1000),raw_detection_index=i) for i in range(110)]
        merged,_=ev.merge_for_threshold(objs,[],.001);self.assertEqual(len(merged),100);self.assertGreaterEqual(min(x['confidence_score'] for x in merged),.21)
    def test_selection_guardrail(self):
        b={'AP_small':.1,'AP50_95':.3}
        self.assertTrue(ev.eligible(b,{'AP_small':.11,'AP50_95':.29}))
        self.assertFalse(ev.eligible(b,{'AP_small':.1,'AP50_95':.4}))
        self.assertFalse(ev.eligible(b,{'AP_small':.2,'AP50_95':.28}))
    def test_test_split_rejected_before_checkpoint_open(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'data').mkdir();(p/'base').mkdir();(p/'model.pt').write_bytes(b'x')
            for name in ['EVALUATION_MANIFEST.json','METRICS.json']:(p/'base'/name).write_text(json.dumps({'status':'complete','split':'test'}))
            (p/'base/DATASET_SNAPSHOT.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'test is forbidden'):ev.preflight(p/'data',p/'base',p/'model.pt')
    def test_input_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'input';p.write_text('original');proof=ev.fingerprint(p);p.write_text('changed')
            with self.assertRaises(ValueError):ev.check_inputs([proof])

if __name__=='__main__':unittest.main()
