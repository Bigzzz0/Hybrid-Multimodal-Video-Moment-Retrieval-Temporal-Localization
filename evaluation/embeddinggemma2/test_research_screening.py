import copy
import unittest
import numpy as np
from .research_screening import cross_scene, predictions, semantic_scores, timeline, unit
from .research_vpc import padded_box
from .research_reference_calibration import robust_scale

class ResearchScreeningTests(unittest.TestCase):
    def test_scores_are_finite_when_text_covariance_is_rank_deficient(self):
        normal=np.array([[1.,0,0],[1.,0,0]])
        groups=[np.array([[0.,1,0],[0.,1,0]]),np.array([[0.,0,1]])]
        result=semantic_scores(np.eye(3),normal,groups)
        self.assertTrue(all(np.isfinite(v).all() for v in result.values()))
        self.assertGreater(result['semantic_margin'][1],result['semantic_margin'][0])
        self.assertTrue(np.all((result['flashback_small_bank']>=0)&(result['flashback_small_bank']<=1)))

    def test_held_scene_labels_do_not_choose_its_threshold(self):
        rows=[];data={}
        for scene in ('a','b','c'):
            for number in range(2):
                key=scene+str(number)
                rows.append({'id':key,'scene_id':scene,'dataset':'toy','recording_id':key,'duration':3.,
                             'events':[{'t_start':1.,'t_end':2.}] if number else []})
                data[key]=(np.arange(4),np.array([.1,.9 if number else .2,.1]))
        first=cross_scene(rows,data)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['scene_id']=='a':r['events']=[]
        second=cross_scene(changed,data)
        self.assertEqual(first['folds'][0]['threshold'],second['folds'][0]['threshold'])
        self.assertFalse(set(first['folds'][0]['held_ids'])&set(first['folds'][0]['calibration_ids']))

    def test_temporal_boundaries_do_not_bridge_negative_gap(self):
        result=predictions(np.arange(6),np.array([1,1,0,1,1]),.5)
        self.assertEqual([(r['t_start'],r['t_end']) for r in result],[(0.,2.),(3.,5.)])

    def test_tail_and_nonfinite_rejection(self):
        edges,scores=timeline([1.],[(0.,1.2)],1.2)
        self.assertEqual(edges.tolist(),[0.,1.,1.2])
        with self.assertRaises(ValueError):timeline([float('nan')],[(0.,1.)],1.)
        with self.assertRaises(ValueError):unit([[0.,0.]])

    def test_object_crop_keeps_context_and_stays_on_canvas(self):
        x1,y1,x2,y2=padded_box([1,1,20,40],100,100)
        self.assertTrue(0<=x1<=1 and 0<=y1<=1 and x2>=21 and y2>=41)
        self.assertTrue(x2<=100 and y2<=100)

    def test_constant_normal_channel_is_disabled_not_amplified(self):
        values=np.array([[100.,2.]])
        normal=np.array([[0.,1.],[0.,2.],[0.,3.]])
        scaled,available=robust_scale(values,normal)
        self.assertEqual(available.tolist(),[False,True])
        self.assertEqual(scaled[0,0],0)
        self.assertTrue(np.isfinite(scaled).all())

if __name__=='__main__':unittest.main()
