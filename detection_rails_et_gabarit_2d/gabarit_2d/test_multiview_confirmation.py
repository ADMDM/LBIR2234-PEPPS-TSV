"""Pure confirmation contracts; no model, inference, or accuracy claim."""
import copy
import math
import unittest
import numpy as np
from multiview_confirmation import filter_confirmed_tiles


def candidate(ident='t0002', view='tile_0000', raw=2):
    mask=np.zeros((20,20),bool);mask[5:12,5:12]=1;mask[7:9,7:9]=0
    return dict(candidate_id=ident,view_id=view,view_kind='tile',model_source='litter',
                raw_detection_index=raw,mask=mask,confidence_score=.78)


def evidence(**changes):
    row=dict(rule='cross_view_mask_iou',kept_candidate_id='t0002',kept_view_id='tile_0000',
             kept_raw_detection_index=2,suppressed_candidate_id='t0008',
             suppressed_view_id='tile_0001',suppressed_raw_detection_index=8,mask_iou=.7,mask_ios=.8)
    row.update(changes)
    return row


class ConfirmationTests(unittest.TestCase):
    def test_confirmation_preserves_dictionary_mask_holes_and_score(self):
        obj=candidate();before=copy.deepcopy(obj);ev=evidence();old=copy.deepcopy(ev)
        kept,audit=filter_confirmed_tiles([obj],[ev])
        self.assertIs(kept[0],obj);np.testing.assert_array_equal(obj['mask'],before['mask'])
        self.assertEqual(obj['confidence_score'],before['confidence_score']);self.assertEqual(ev,old)
        self.assertTrue(audit[0]['confirmed']);self.assertEqual(audit[0]['distinct_tile_view_count'],2)

    def test_iou_threshold_is_strict(self):
        for iou,wanted in [(.49,False),(.5,False),(.50001,True)]:
            kept,audit=filter_confirmed_tiles([candidate()],[evidence(mask_iou=iou)])
            self.assertEqual(bool(kept),wanted)

    def test_same_view_global_view_and_ios_only_cannot_confirm(self):
        for change in [dict(suppressed_view_id='tile_0000'),dict(suppressed_view_id='global'),
                       dict(suppressed_view_id='baseline'),dict(rule='cross_view_truncated_mask_ios',mask_ios=1)]:
            with self.subTest(change=change):
                kept,audit=filter_confirmed_tiles([candidate()],[evidence(**change)])
                self.assertFalse(kept);self.assertEqual(audit[0]['distinct_tile_view_count'],1)

    def test_duplicate_votes_from_same_view_count_once(self):
        rows=[evidence(),evidence(),evidence(suppressed_candidate_id='t0009',suppressed_raw_detection_index=9)]
        kept,audit=filter_confirmed_tiles([candidate()],rows)
        self.assertEqual(len(kept),1);self.assertEqual(audit[0]['distinct_tile_view_count'],2)
        self.assertEqual(len(audit[0]['supporting_evidence']),3)

    def test_three_actual_views_count_three(self):
        rows=[evidence(),evidence(suppressed_view_id='tile_0002',suppressed_candidate_id='t0010',suppressed_raw_detection_index=10)]
        _,audit=filter_confirmed_tiles([candidate()],rows)
        self.assertEqual(audit[0]['distinct_tile_view_count'],3)

    def test_kept_id_mixup_with_matching_raw_view_rejected(self):
        with self.assertRaisesRegex(ValueError,'Kept candidate ID'):
            filter_confirmed_tiles([candidate()],[evidence(kept_candidate_id='foreign')])

    def test_kept_view_or_raw_mismatch_rejected(self):
        for change in [dict(kept_view_id='tile_0002'),dict(kept_raw_detection_index=3)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                filter_confirmed_tiles([candidate()],[evidence(**change)])

    def test_suppressed_survivor_identity_rejected(self):
        for change in [dict(suppressed_candidate_id='t0002'),dict(suppressed_raw_detection_index=2)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                filter_confirmed_tiles([candidate()],[evidence(**change)])

    def test_suppressed_identity_cannot_vote_under_multiple_views(self):
        rows=[evidence(),evidence(suppressed_view_id='tile_0002')]
        with self.assertRaisesRegex(ValueError,'identity is inconsistent'):
            filter_confirmed_tiles([candidate()],rows)

    def test_unrelated_audit_does_not_confirm(self):
        kept,audit=filter_confirmed_tiles([candidate()],
            [evidence(kept_candidate_id='t0099',kept_raw_detection_index=99)])
        self.assertFalse(kept);self.assertFalse(audit[0]['supporting_evidence'])

    def test_nonfinite_or_invalid_iou_and_indices_rejected(self):
        for change in [dict(mask_iou=math.nan),dict(mask_iou=math.inf),dict(mask_iou=1.1),
                       dict(mask_iou=True),dict(kept_raw_detection_index=-1),
                       dict(suppressed_raw_detection_index=2.5)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                filter_confirmed_tiles([candidate()],[evidence(**change)])

    def test_optional_raw_indices_and_suppressed_id_can_be_absent(self):
        obj=candidate();obj.pop('raw_detection_index');row=evidence()
        for key in ['kept_raw_detection_index','suppressed_raw_detection_index','suppressed_candidate_id']:
            row.pop(key)
        self.assertEqual(len(filter_confirmed_tiles([obj],[row])[0]),1)

    def test_duplicate_candidate_id_and_raw_index_rejected(self):
        for second in [candidate(),candidate('t0003','tile_0002',2)]:
            with self.assertRaises(ValueError):
                filter_confirmed_tiles([candidate(),second],[])

    def test_non_tile_and_inconsistent_tile_index_rejected(self):
        for change in [dict(view_id='global'),dict(view_kind='global'),dict(tile_index=3)]:
            obj=candidate();obj.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                filter_confirmed_tiles([obj],[])

    def test_output_order_is_input_order(self):
        a=candidate();b=candidate('t0012','tile_0002',12)
        evb=evidence(kept_candidate_id='t0012',kept_view_id='tile_0002',kept_raw_detection_index=12,
                     suppressed_candidate_id='t0020',suppressed_view_id='tile_0003',suppressed_raw_detection_index=20)
        kept,_=filter_confirmed_tiles([b,a],[evidence(),evb])
        self.assertEqual([o['candidate_id'] for o in kept],['t0012','t0002'])

    def test_empty_cases(self):
        self.assertEqual(filter_confirmed_tiles([],[]),([],[]))
        self.assertFalse(filter_confirmed_tiles([candidate()],[])[0])


if __name__=='__main__':unittest.main()
