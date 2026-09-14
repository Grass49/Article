"""Software tests only: generated images/fake inference are NOT research results."""
import argparse, contextlib, io, json, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import validate_onnx_om as v

class MetricsTest(unittest.TestCase):
    def test_equal_and_scale(self):
        r = np.array([[1.,2.,3.],[2.,1.,0.]])
        rp, dp, c, e = v.metrics(r,r.copy())
        np.testing.assert_array_equal(rp,dp)
        np.testing.assert_allclose(c,1)
        np.testing.assert_allclose(e,0)
        _,_,c,e = v.metrics(r,10*r)
        np.testing.assert_allclose(c,1)
        np.testing.assert_allclose(e,9)

    def test_class_flip_despite_high_cosine(self):
        rp,dp,c,e = v.metrics(np.array([[1.,1.001]]),np.array([[1.001,1.]]))
        self.assertNotEqual(rp[0],dp[0])
        self.assertGreater(c[0],.999)
        self.assertGreater(e[0],0)

    def test_zero_nonfinite_and_shape(self):
        _,_,c,e=v.metrics(np.zeros((1,3)),np.zeros((1,3)))
        self.assertTrue(np.isnan(c[0]))
        self.assertEqual(e[0],0)
        with self.assertRaises(ValueError):
            v.metrics(np.array([[np.nan]]),np.array([[1.]]))
        with self.assertRaises(ValueError):
            v.classification_matrix(np.zeros(8),2,4)

    def test_padding(self):
        x=np.arange(15).reshape(5,3)
        parts=list(v.chunks(x,4))
        self.assertEqual([valid for _,valid,_ in parts],[4,1])
        np.testing.assert_array_equal(parts[1][2],np.repeat(x[4:],4,axis=0))

class PipelineTest(unittest.TestCase):
    def test_prepare_run_compare_and_stale_pair_protection(self):
        with tempfile.TemporaryDirectory(prefix="c6_software_test_") as td:
            base=Path(td); images=base/"images";images.mkdir()
            for i in range(9):
                folder=images/str(i%3);folder.mkdir(exist_ok=True)
                Image.new("RGB",(300+i,270),(i*20,60+i,110)).save(folder/(str(i)+".png"))
            fake_model=base/"mock.onnx";fake_model.write_bytes(b"software-test-not-a-real-model")
            job=base/"job"; run=base/"run"
            class Session:
                def __init__(self,*a,**kw):pass
                def get_inputs(self):
                    return [types.SimpleNamespace(name="input",shape=[1,3,224,224],type="tensor(float)")]
                def get_outputs(self):
                    return [types.SimpleNamespace(name="output")]
                def run(self,names,feeds):
                    x=feeds["input"]
                    means=x.mean(axis=(2,3))
                    return [means.astype(np.float32)]
            fake_ort=types.SimpleNamespace(InferenceSession=Session,__version__="FAKE_TEST_ONLY")
            args=v.parser().parse_args(["prepare","--images",str(images),"--onnx",str(fake_model),
                "--job",str(job),"--samples","5","--batch-size","4","--classes","3"])
            with patch.dict(sys.modules,{"onnxruntime":fake_ort}),contextlib.redirect_stdout(io.StringIO()):
                v.prepare(args)
            m=v.read_json(job/"manifest.json")
            self.assertEqual(m["reference_batch"],1)
            self.assertEqual([b["valid"] for b in m["batches"]],[4,1])
            chosen=v.read_json(job/"samples.json")["samples"]
            reused,_=v.choose_samples(images,100,999,"random",job/"samples.json")
            self.assertEqual(chosen,reused)
            np.testing.assert_equal(len({r["sha256"] for r in chosen}),5)
            om=base/"mock.om";om.write_bytes(b"software-test-not-an-OM-model")
            def fake_msame(cmd,**kw):
                self.assertEqual(cmd[cmd.index("--loop")+1],"1")
                inp=Path(cmd[cmd.index("--input")+1])
                output=Path(cmd[cmd.index("--output")+1])/"timestamp"
                output.mkdir()
                x=np.fromfile(inp,dtype=np.float32).reshape(4,3,224,224)
                y=x.mean(axis=(2,3)).astype(np.float32)
                y.tofile(output/"batch_output_0.bin")
                return types.SimpleNamespace(returncode=0)
            args=v.parser().parse_args(["run-om","--job",str(job),"--om",str(om),
                "--msame",sys.executable,"--run",str(run)])
            with patch.object(v.subprocess,"run",side_effect=fake_msame),contextlib.redirect_stdout(io.StringIO()):
                v.run_om(args)
            output=np.load(run/"om_outputs.npy",allow_pickle=False)
            self.assertEqual(output.shape,(5,3))
            args=v.parser().parse_args(["compare","--job",str(job),"--run",str(run)])
            with contextlib.redirect_stdout(io.StringIO()):
                v.compare(args)
            s=v.read_json(run/"report/summary.json")
            self.assertEqual(s["sample_count"],5)
            self.assertEqual(s["matching_top1_count"],5)
            self.assertFalse(s["padded_rows_counted"])
            self.assertLess(s["relative_l2_error"]["max"],1e-6)
            with self.assertRaises(ValueError):
                v.compare(args)  # refuse silent overwrite
            # Changing pairing provenance is rejected before output is used.
            m["om_batch"]=16;v.save_json(job/"manifest.json",m)
            with self.assertRaisesRegex(ValueError,"Wrong job/run"):
                v.compare(args)

    def test_wrong_om_binary_size_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c6_size_test_") as td:
            base=Path(td);job=base/"job";job.mkdir()
            x=job/"input.bin";x.write_bytes(b"test")
            samples=job/"samples.json";v.save_json(samples,{"samples":[]})
            v.save_json(job/"manifest.json",dict(sample_manifest_sha256=v.sha(samples),
                om_batch=4,classes=1000,output_kind="logits",sample_count=1,
                batches=[dict(index=0,start=0,valid=1,input_file="input.bin",sha256=v.sha(x))]))
            om=base/"mock.om";om.write_bytes(b"test")
            def fake(cmd,**kw):
                out=Path(cmd[cmd.index("--output")+1])
                (out/"output_0.bin").write_bytes(b"wrong-size")
                return types.SimpleNamespace(returncode=0)
            a=v.parser().parse_args(["run-om","--job",str(job),"--om",str(om),
                "--msame",sys.executable,"--run",str(base/"run")])
            with patch.object(v.subprocess,"run",side_effect=fake):
                with self.assertRaisesRegex(ValueError,"byte count"):
                    v.run_om(a)
            self.assertEqual(v.read_json(base/"run/run_manifest.json")["status"],"failed")

if __name__=="__main__":
    unittest.main(verbosity=2)
