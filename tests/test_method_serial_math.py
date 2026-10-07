"""Synthetic CPU tests only: no environment, checkpoint, planner or optimizer."""
import math
import unittest
import torch
from torch.nn import functional as F
from cp_disr.blocksworld.method_serial.heads_math import (
    NextGoalActionHead, loss_terms_v3, label_information,
    calibrated_logits, calibration_objectives,
)


def fixture():
    torch.manual_seed(9)
    B,G,K,D = 3,3,4,128
    z = torch.randn(B,G,D)
    zn = torch.randn(B,K,G,D)
    t = torch.zeros(B,G)
    tn = torch.zeros(B,K,G)
    gv = torch.ones(B,G,dtype=torch.bool)
    pending = gv.clone()
    legal = torch.tensor([[1,1,1,0],[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
    base = torch.randn(B,K).masked_fill(~legal,-torch.inf)
    optimal = torch.tensor([[1,1,0,0],[1,1,0,0],[1,0,0,0]],dtype=torch.bool)
    d = torch.where(optimal,torch.tensor(3.),torch.tensor(5.))
    # Row0: one event, every optimal action. Row1: informative diagonal.
    # Row2: UNKNOWN; deliberately empty original target.
    y = torch.zeros(B,G,K,dtype=torch.bool)
    y[0,0,:2] = True
    y[1,0,0] = True
    y[1,1,1] = True
    complete = torch.tensor([1,1,0],dtype=torch.bool)
    return z,zn,t,tn,base,gv,pending,legal,optimal,d,y,complete


class MathTests(unittest.TestCase):
    def forward(self,nonzero=False):
        f=fixture()
        model=NextGoalActionHead()
        if nonzero:
            with torch.no_grad():
                model.gate[-1].weight.normal_(std=.02)
                model.residual[-1].weight.normal_(std=.02)
        out=model(*f[:8])
        return model,out,f

    def terms(self,out,f,mode):
        return loss_terms_v3(out,f[8],f[9],f[7],f[6],f[10],f[11],mode)

    def test_01_initial_policy_and_parameter_count(self):
        model,out,f=self.forward()
        self.assertEqual(sum(p.numel() for p in model.parameters()),82690)
        torch.testing.assert_close(out.log_action.exp(),F.softmax(f[4],-1))

    def test_02_pair_information_not_event_information(self):
        _,_,f=self.forward()
        info=label_information(f[10],f[8],f[6],f[11])
        self.assertEqual(info["pair_information"].tolist(),[False,True,False])
        self.assertEqual(info["singleton_event_rectangle"].tolist(),[True,False,False])

    def test_03_singleton_rectangle_fact_joint_identical(self):
        _,out,f=self.forward(True)
        a=self.terms(out,f,"FACT")["loss"][0]
        b=self.terms(out,f,"JOINT")["loss"][0]
        torch.testing.assert_close(a,b)

    def test_04_true_pair_target_is_stricter(self):
        _,out,f=self.forward(True)
        a=self.terms(out,f,"FACT")
        b=self.terms(out,f,"JOINT")
        self.assertGreater(float((b["loss"][1]-a["loss"][1]).detach()),0.)
        self.assertGreater(float(b["pair_specific"][1].detach()),0.)

    def test_05_unknown_no_auxiliary_loss(self):
        _,out,f=self.forward(True)
        for mode in ("FACT","JOINT"):
            torch.testing.assert_close(self.terms(out,f,mode)["loss"][2],
                                       self.terms(out,f,"BASE")["loss"][2])

    def test_06_full_cartesian_base_fact_joint_equal(self):
        _,out,f=self.forward(True)
        f=list(f)
        f[10]=f[6][:,:,None]&f[8][:,None,:]
        f[11]=torch.ones(3,dtype=torch.bool)
        a=self.terms(out,f,"BASE")["loss"]
        for mode in ("FACT","JOINT"):
            torch.testing.assert_close(a,self.terms(out,f,mode)["loss"])

    def test_07_finite_backward_all_three(self):
        for mode in ("BASE","FACT","JOINT"):
            model,out,f=self.forward(True)
            loss=self.terms(out,f,mode)["loss"].mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_08_goal_permutation(self):
        model,out,f=self.forward(True)
        z,zn,t,tn,b,gv,p,l=f[:8]
        order=torch.tensor([2,0,1])
        shuffled=model(z[:,order],zn[:,:,order],t[:,order],tn[:,:,order],
                       b,gv[:,order],p[:,order],l)
        torch.testing.assert_close(out.log_action,shuffled.log_action)

    def test_09_candidate_permutation(self):
        model,out,f=self.forward(True)
        z,zn,t,tn,b,gv,p,l=f[:8]
        order=torch.tensor([2,0,3,1])
        shuffled=model(z,zn[:,order],t,tn[:,order],b[:,order],gv,p,l[:,order])
        torch.testing.assert_close(out.log_action[:,order],shuffled.log_action)

    def test_10_affine_temperature_preserves_logits(self):
        v=torch.tensor([8.,-3.])
        vn=torch.tensor([[5.,9.,2.],[-1.,-8.,4.]])
        legal=torch.ones_like(vn,dtype=torch.bool)
        a0=.031
        b=calibrated_logits(v,vn,legal,a0,37.,torch.tensor(math.log(1/a0)))
        torch.testing.assert_close(b,v[:,None]-vn,atol=2e-4,rtol=2e-5)

    def test_11_relative_offset_invariance(self):
        w=torch.tensor([5.,4.2,1.9,0.1])
        d=torch.tensor([5.,4.,2.,0.])
        pairs=torch.tensor([[0,1],[1,2],[2,3]])
        a=calibration_objectives(w,d,pairs)
        b=calibration_objectives(w+10,d,pairs)
        torch.testing.assert_close(a["relative"],b["relative"],atol=1e-6,rtol=1e-6)
        self.assertGreater(float(b["absolute"]),float(a["absolute"]))

    def test_12_old_consistency_residual_is_error_difference(self):
        w=torch.tensor([8.,3.,5.])
        d=torch.tensor([6.,2.,4.])
        i=torch.tensor([0,0]);j=torch.tensor([1,2])
        residual=(w[i]-w[j])-(d[i]-d[j])
        e=w-d
        torch.testing.assert_close(residual,e[i]-e[j])

    def test_13_singleton_fact_joint_gradients_identical(self):
        grads=[]
        for mode in ("FACT","JOINT"):
            model,out,f=self.forward(True)
            self.terms(out,f,mode)["loss"][0].backward()
            grads.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()]))
        torch.testing.assert_close(grads[0],grads[1])

    def test_14_invalid_complete_projection_rejected(self):
        _,out,f=self.forward()
        f=list(f)
        f[10]=f[10].clone();f[10][0,0,1]=False
        with self.assertRaises(ValueError):
            self.terms(out,f,"JOINT")


if __name__=="__main__":
    unittest.main()
