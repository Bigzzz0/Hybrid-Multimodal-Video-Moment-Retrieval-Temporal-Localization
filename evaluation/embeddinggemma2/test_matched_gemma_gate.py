import copy
import unittest
from .compare_matched_gemma_gate import filter_payload,predictions

class MatchedGateTests(unittest.TestCase):
    def test_gate_only_removes_units_and_keeps_frames_prompts_and_references(self):
        payload=[{'id':'a','scene_context':'fixed','frame_paths':['one','two'],'timestamps':[0,1],
                  'reference_frames':[{'path':'normal'}]}, {'id':'b','scene_context':'fixed','frame_paths':['three'],'timestamps':[2]}]
        before=copy.deepcopy(payload)
        intervals={'a':{'video_id':'v','t_start':0,'t_end':12},'b':{'video_id':'v','t_start':12,'t_end':24}}
        result=filter_payload(payload,intervals,{'v':[{'t_start':0,'t_end':12}]})
        self.assertEqual(result,[payload[0]])
        self.assertEqual(payload,before)
        result[0]['frame_paths'].append('changed')
        self.assertEqual(payload,before)

    def test_skipped_video_is_explicit_and_failed_verification_is_unknown(self):
        rows=[{'id':'v'},{'id':'skipped'}]
        result=predictions(rows,[{'id':'a','status':'unverified','verdict':None}],
                            {'a':{'video_id':'v','t_start':0,'t_end':12}})
        self.assertEqual(result['skipped'],[])
        self.assertEqual(result['v'][0]['assessment'],'uncertain')
        with self.assertRaises(ValueError):predictions(rows,[{'id':'a'},{'id':'a'}],{})

if __name__=='__main__':unittest.main()
