import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import transformer_labkit as kit

class GeneratorTests(unittest.TestCase):
    def test_all_240_assignments(self):
        with tempfile.TemporaryDirectory() as d:
            for lab in range(1,5):
                for v in range(1,61):
                    with self.subTest(lab=lab,variant=v):
                        p=Path(d)/f'{lab}_{v}'
                        kit.generate(lab,v,p,n=4)
                        self.assertTrue(kit.verify(p))

    def test_all_table_parameters(self):
        for v in range(1,61):
            c=kit.parameters(v,2)
            self.assertEqual(c['table_seed'],1000+7919*v)
            self.assertEqual(c['max_length'],16+8*((v-1)%6))
            self.assertEqual(c['vocab_size'],32+16*((v-1)%5))
            self.assertAlmostEqual(c['noise'],.05*((v-1)%4))
        with self.assertRaises(ValueError):kit.parameters(0,1)
        with self.assertRaises(ValueError):kit.parameters(61,1)
        with self.assertRaises(ValueError):kit.parameters(1,5)

    def test_order_and_role_independence(self):
        c=kit.parameters(5,2)
        a=[kit.sample(c,'train',i) for i in range(8)]
        b={i:kit.sample(c,'train',i) for i in reversed(range(8))}
        self.assertEqual(a,[b[i] for i in range(8)])
        kit.rng(2,5,'extra-diagnostic').normal(size=100)
        self.assertEqual(a[0],kit.sample(c,'train',0))
        self.assertNotEqual(kit.seed(2,5,'train'),kit.seed(2,5,'test'))

    def test_repeat_logical_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            a=kit.generate(2,17,Path(d)/'a',8)
            b=kit.generate(2,17,Path(d)/'b',8)
            self.assertEqual(a['logical_arrays'],b['logical_arrays'])
            for split in kit.SPLITS:
                self.assertEqual((Path(d)/'a'/f'{split}.jsonl').read_bytes(),(Path(d)/'b'/f'{split}.jsonl').read_bytes())
            with self.assertRaises(FileExistsError):kit.generate(2,17,Path(d)/'a',8)

    def test_semantic_mutation_even_with_new_checksum(self):
        for v in range(1,7):
            c=kit.parameters(v,2);row=kit.sample(c,'train',0)
            i=row['loss_mask'].index(1)
            row['target_ids'][i]=(row['target_ids'][i]+1)%c['vocab_size']
            row['checksum']=kit.sha(kit.canonical({k:v for k,v in row.items() if k!='checksum'}))
            with self.assertRaisesRegex(ValueError,'semantic oracle'):kit.validate(row,c)

    def test_corrupted_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';kit.generate(2,1,p,4)
            with (p/'train.jsonl').open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'file checksum'):kit.verify(p)

    def test_invalid_masks_rejected(self):
        c=kit.parameters(1,2);row=kit.sample(c,'train',0)
        row['loss_mask'][-1]=1
        row['checksum']=kit.sha(kit.canonical({k:v for k,v in row.items() if k!='checksum'}))
        with self.assertRaises(ValueError):kit.validate(row,c)

    def test_listings_numpy(self):
        spec=importlib.util.spec_from_file_location('attention_example',ROOT/'listings/numpy_attention.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.checks()

    def test_all_python_syntax(self):
        for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))

@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed; optional integration test skipped')
class TorchTests(unittest.TestCase):
    def test_contract_gradient_and_checkpoint(self):
        import torch
        import importlib.util
        spec=importlib.util.spec_from_file_location('fragments',ROOT/'listings/torch_fragments.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.check_batch(torch.ones((2,3),dtype=torch.long),torch.ones((2,3),dtype=torch.bool))
        param=torch.tensor([.3],dtype=torch.float64,requires_grad=True)
        (param**2).sum().backward();mod.check_gradient(param,0,lambda:(param**2).sum(),param.grad[0].item())
        model=torch.nn.Linear(2,2).double();opt=torch.optim.AdamW(model.parameters());sched=torch.optim.lr_scheduler.StepLR(opt,1)
        class Adapter(torch.nn.Module):
            def __init__(self,m):super().__init__();self.m=m
            def forward(self,x,mask):return self.m(x)
        adapter=Adapter(model)
        mod.training_step(adapter,opt,sched,torch.nn.CrossEntropyLoss(),torch.ones((2,2),dtype=torch.float64),None,torch.tensor([0,1]),torch.ones(2,dtype=torch.bool),1.)
        with tempfile.TemporaryDirectory() as d:
            mod.save_checkpoint(model,opt,sched,1,{},Path(d)/'temp.pt',Path(d)/'state.pt',{})
            self.assertTrue((Path(d)/'state.pt').exists())

if __name__=='__main__':unittest.main(verbosity=2)
