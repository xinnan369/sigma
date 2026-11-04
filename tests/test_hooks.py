import torch
import pytest
from experiments.exp0.hooks import CaptureOutput,ReplaceOutput


def test_capture_output_tensor_and_detached_copy():
    m=torch.nn.Linear(3,2,bias=False).eval(); cap=CaptureOutput(m)
    x=torch.ones(1,3)
    y=m(x); cap.close()
    assert len(cap.outputs)==1 and torch.allclose(cap.outputs[0],y)
    assert cap.outputs[0] is not y


def test_replace_output_changes_output_not_model_weights():
    m=torch.nn.Linear(3,2,bias=False).eval(); w=m.weight.detach().clone()
    replace=ReplaceOutput(m,torch.ones(1,2)*77)
    y=m(torch.ones(1,3));replace.close()
    assert replace.calls==1 and torch.allclose(y,torch.ones(1,2)*77)
    assert torch.equal(w,m.weight)


def test_tuple_output_replacement_and_shape_guard():
    class Two(torch.nn.Module):
        def forward(self,x): return x,2*x
    m=Two();repl=ReplaceOutput(m,(torch.zeros(1,2),torch.ones(1,2)))
    a,b=m(torch.ones(1,2));repl.close()
    assert torch.all(a==0) and torch.all(b==1)
    wrong=ReplaceOutput(m,(torch.zeros(1,3),torch.ones(1,2)))
    with pytest.raises(ValueError):m(torch.ones(1,2))
    wrong.close()
