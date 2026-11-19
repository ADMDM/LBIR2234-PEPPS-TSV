import tempfile
from pathlib import Path
import json
import unittest
import evaluate_selective_validation as ev


class SelectiveProtocolTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(AP_small=.1, AP50_95=.35)
        self.op = dict(precision=.52, small_recall=.25, F2=.51)
        self.new = dict(AP_small=.11, AP50_95=.35)
        self.new_op = dict(precision=.51, small_recall=.3, F2=.54)

    def test_each_guardrail_is_required(self):
        ok, checks = ev.eligible(self.base, self.op, self.new, self.new_op)
        self.assertTrue(ok); self.assertTrue(all(checks.values()))
        cases = [('precision', .48), ('small_recall', .25), ('F2', .50)]
        for key, value in cases:
            changed = dict(self.new_op, **{key: value})
            ok, checks = ev.eligible(self.base, self.op, self.new, changed)
            self.assertFalse(ok); self.assertFalse(checks[key])
        for key, value in [('AP_small', .1), ('AP50_95', .33)]:
            ok, checks = ev.eligible(self.base, self.op, dict(self.new, **{key: value}), self.new_op)
            self.assertFalse(ok); self.assertFalse(checks[key])

    def test_no_eligible_configuration_is_not_forced(self):
        self.assertIsNone(ev.choose([]))
        self.assertIsNone(ev.choose([dict(eligible=False, operating=self.new_op, grid_index=0)]))

    def test_selection_prefers_small_recall_then_precision_then_frozen_order(self):
        a=dict(eligible=True,grid_index=0,operating=dict(small_recall=.3,precision=.9))
        b=dict(eligible=True,grid_index=1,operating=dict(small_recall=.4,precision=.5))
        c=dict(eligible=True,grid_index=2,operating=dict(small_recall=.4,precision=.6))
        d=dict(eligible=True,grid_index=3,operating=dict(small_recall=.4,precision=.6))
        self.assertIs(ev.choose([a,b,c,d]),c)
        self.assertIs(ev.choose([d,c,b,a]),c)

    def test_test_is_rejected_before_any_cache_or_checkpoint_read(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'EVALUATION_MANIFEST.json').write_text(json.dumps(dict(status='complete',split='test')))
            with self.assertRaisesRegex(ValueError,'test forbidden'):ev.preflight(p)

    def test_metric_cap_is_reported_without_deleting_input_predictions(self):
        ps=[dict(image_id=1) for _ in range(102)]+[dict(image_id=2)]
        self.assertEqual(ev.truncation_counts(ps),dict(images_above_max100=1,predictions_beyond_max100=2))
        self.assertEqual(len(ps),103)

    def test_frozen_grid_and_policy_are_explicit(self):
        self.assertEqual(ev.PROTOCOL['confidence_grid'],[.35,.5,.65,.8])
        self.assertEqual(ev.PROTOCOL['area_fraction_grid'],[.001,.002,.004])
        self.assertIn('no replacement',ev.PROTOCOL['baseline_policy'])
        self.assertFalse(ev.PROTOCOL['test_used'])


if __name__=='__main__':unittest.main()
