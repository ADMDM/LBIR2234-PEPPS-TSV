import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import evaluate_consensus_validation as ev


class ConsensusProtocolTests(unittest.TestCase):
    def test_same_five_guardrails_remain_required(self):
        base=dict(AP_small=.10,AP50_95=.35)
        op=dict(precision=.52,small_recall=.25,F2=.51)
        candidate=dict(AP_small=.09,AP50_95=.35)
        candidate_op=dict(precision=.51,small_recall=.30,F2=.53)
        ok,checks=ev.eligible(base,op,candidate,candidate_op)
        self.assertFalse(ok)
        self.assertEqual([k for k,v in checks.items() if not v],['AP_small'])
        candidate['AP_small']=.11
        self.assertTrue(ev.eligible(base,op,candidate,candidate_op)[0])

    def test_confirmation_is_additional_to_the_same_frozen_grid(self):
        self.assertEqual(ev.PROTOCOL['confidence_grid'],[.35,.5,.65,.8])
        self.assertEqual(ev.PROTOCOL['area_fraction_grid'],[.001,.002,.004])
        rule=ev.PROTOCOL['confirmation_rule']
        self.assertIn('mask_iou>.5',rule)
        self.assertIn('globals and IoS-only records do not count',rule)
        self.assertIn('no score or mask changes',rule)
        self.assertFalse(ev.PROTOCOL['test_used'])

    def test_no_configuration_is_chosen_if_all_fail(self):
        self.assertIsNone(ev.choose([dict(eligible=False,grid_index=0)]))

    def test_earlier_experiment_must_really_have_been_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=p/'SELECTION.json';s.write_text(json.dumps(dict(status='complete',split='val',selected_eligible=True,selected_config={'min_confidence':.5})))
            (p/'EVALUATION_MANIFEST.json').write_text(json.dumps(dict(status='complete',selection={'sha256':ev.sha(s)})))
            prior=dict(checkpoint_sha256='x',dataset_sha256='y')
            args=SimpleNamespace(prior_validation=p,rejected_selective=s,out=p/'out')
            with patch.object(ev,'preflight',return_value=(p,prior,[],None,[],[])):
                with self.assertRaisesRegex(ValueError,'was not rejected'):ev.run(args)
            self.assertFalse((p/'out').exists())

    def test_test_split_is_rejected_before_reading_predictions(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'EVALUATION_MANIFEST.json').write_text(json.dumps(dict(status='complete',split='test')))
            with self.assertRaisesRegex(ValueError,'test forbidden'):ev.preflight(p)


if __name__=='__main__':unittest.main()
