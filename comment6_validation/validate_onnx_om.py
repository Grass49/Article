#!/usr/bin/env python3
"""Reviewer Comment 6: paired ONNX/OM validation. No training or search scoring."""
import argparse, csv, hashlib, json, math, platform, random, re, shutil
import subprocess, sys, time
from pathlib import Path
import numpy as np

VERSION = "1.0"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REVIEW_COMMENT = (
"The use of cosine similarity on 10 random inputs is a useful initial check, "
"but it provides limited evidence that model behavior is preserved, especially "
"under FP16 settings. Evaluate a representative held-out sample and report "
"task-level agreement for the classifiers together with one output-error measure. "
"Clarify how outputs were compared across batch sizes and how the 0.999 threshold "
"was selected. Cosine similarity can remain as the admission criterion, but its "
"limitations should be acknowledged."
)
def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def fresh(path):
    path = Path(path).resolve()
    require(not path.exists(), "Output already exists; choose a new directory: " + str(path))
    path.mkdir(parents=True)
    return path

def inside(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    require(p == root or root in p.parents, "Path leaves job directory")
    return p

def csv_write(path, rows):
    require(len(rows) > 0, "No rows to write")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def preprocess(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # torchvision Resize(256) semantics for a PIL image: short edge 256.
        nw, nh = (256, int(256 * h / w)) if w <= h else (int(256 * w / h), 256)
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        left, top = int(round((nw - 224) / 2)), int(round((nh - 224) / 2))
        im = im.crop((left, top, left + 224, top + 224))
        x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)

def choose_samples(root, count, seed, mode, sample_list=None):
    root = Path(root).resolve()
    if sample_list:
        saved = read_json(sample_list)
        chosen = saved["samples"]
        require(len(chosen) > 0, "Empty saved sample list")
        for row in chosen:
            p = inside(root, row["image_id"])
            require(p.is_file() and sha(p) == row["sha256"],
                    "Saved image missing or changed: " + row["image_id"])
        require(len({r["image_id"] for r in chosen}) == len(chosen), "Duplicate image IDs")
        require(len({r["sha256"] for r in chosen}) == len(chosen), "Duplicate image bytes")
        return chosen, "reused fixed sample manifest"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    require(count > 0 and len(files) >= count, "Not enough image files")
    rng = random.Random(seed)
    if mode == "balanced":
        groups = {}
        for p in files:
            groups.setdefault(p.parent.relative_to(root).as_posix(), []).append(p)
        for paths in groups.values():
            rng.shuffle(paths)
        names = sorted(groups)
        rng.shuffle(names)
        ordered = []
        while names:
            next_names = []
            for name in names:
                ordered.append(groups[name].pop())
                if groups[name]:
                    next_names.append(name)
            names = next_names
    else:
        ordered = files[:]
        rng.shuffle(ordered)
    samples, seen = [], set()
    for p in ordered:
        digest = sha(p)
        if digest in seen:
            continue
        seen.add(digest)
        samples.append(dict(image_id=p.relative_to(root).as_posix(),
                            sha256=digest, source_folder=p.parent.relative_to(root).as_posix()))
        if len(samples) == count:
            break
    require(len(samples) == count, "Not enough byte-distinct images")
    return samples, mode + " by folder; folders are not verified class labels"

def chunks(x, batch):
    for start in range(0, len(x), batch):
        valid = min(batch, len(x) - start)
        y = x[start:start + valid]
        if valid < batch:
            y = np.concatenate([y, np.repeat(y[-1:], batch - valid, axis=0)], axis=0)
        yield start, valid, np.ascontiguousarray(y)

def classification_matrix(y, batch, classes):
    y = np.asarray(y)
    require(y.ndim >= 2 and y.shape[0] == batch, "Output must retain the batch dimension")
    require(y.size == batch * classes, "Output is not batch x class scores")
    y = y.reshape(batch, classes)
    require(np.isfinite(y).all(), "NaN/Inf in outputs")
    return y

def prepare(a):
    import onnxruntime as ort
    import PIL
    require(a.batch_size > 0, "Batch size must be positive")
    root = Path(a.images).resolve()
    require(root.is_dir(), "Image directory missing")
    session = ort.InferenceSession(str(Path(a.onnx).resolve()),
                                   providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    require(len(inputs) == 1, "This script supports a single-image-tensor input only")
    inp = inputs[0]
    shape = inp.shape
    require(len(shape) == 4, "Expected NCHW input")
    for got, expected in zip(shape[1:], (3, 224, 224)):
        require(not isinstance(got, int) or got == expected,
                "Expected NCHW [B,3,224,224]; graph layout/preprocessing must be checked")
    require(inp.type in ("tensor(float)", "tensor(float16)"), "Only float32/float16 ONNX input supported")
    if isinstance(shape[0], int) and shape[0] > 0:
        ref_batch = shape[0]
        require(a.ref_batch is None or a.ref_batch == ref_batch, "ONNX fixed batch does not match --ref-batch")
    else:
        ref_batch = a.ref_batch or a.batch_size
    require(ref_batch > 0, "Reference batch must be positive")
    outputs = session.get_outputs()
    require(a.output_name or len(outputs) == 1, "Multiple ONNX outputs: choose --output-name")
    out_name = a.output_name or outputs[0].name
    require(out_name in [o.name for o in outputs], "Unknown ONNX output name")
    samples, sampling = choose_samples(root, a.samples, a.seed, a.sampling, a.sample_list)
    # Preprocess once; both models receive numerically identical input values.
    x = np.stack([preprocess(inside(root, r["image_id"])) for r in samples])
    om_dtype = np.dtype(a.om_input_dtype)
    x = x.astype(om_dtype).astype(np.float32)
    ref_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
    if ref_dtype == np.float16:
        require(om_dtype == np.dtype("float16"),
                "ONNX input is FP16; set --om-input-dtype float16 only if OM interface is also FP16")
    ref = []
    for _, valid, batch in chunks(x, ref_batch):
        y = session.run([out_name], {inp.name: batch.astype(ref_dtype)})[0]
        ref.append(classification_matrix(y, len(batch), a.classes)[:valid].astype(np.float64))
    ref = np.concatenate(ref)
    job = fresh(a.job)
    (job / "inputs").mkdir()
    np.save(job / "reference_outputs.npy", ref, allow_pickle=False)
    batches = []
    for start, valid, batch in chunks(x, a.batch_size):
        relative = "inputs/batch_%05d.bin" % len(batches)
        path = job / relative
        batch.astype(om_dtype).tofile(path)
        batches.append(dict(index=len(batches), start=start, valid=valid,
                            input_file=relative, sha256=sha(path)))
    sample_manifest = dict(seed=a.seed, sampling=sampling, samples=samples)
    save_json(job / "samples.json", sample_manifest)
    csv_write(job / "samples.csv", samples)
    manifest = dict(version=VERSION, dataset=a.dataset_name, dataset_root=str(root),
        sample_manifest_sha256=sha(job / "samples.json"), sample_count=len(samples),
        unique_source_folders=len({r["source_folder"] for r in samples}),
        sampling=sampling, seed=a.seed, onnx_sha256=sha(a.onnx), onnx_path=str(Path(a.onnx).resolve()),
        input_name=inp.name, onnx_input_shape=shape, onnx_input_dtype=inp.type,
        reference_batch=ref_batch, om_batch=a.batch_size, om_input_dtype=om_dtype.name,
        classes=a.classes, output_name=out_name, output_kind=a.output_kind,
        reference_sha256=sha(job / "reference_outputs.npy"), batches=batches,
        preprocessing="RGB; shorter edge 256 bilinear; center crop 224; /255; ImageNet mean/std; NCHW",
        preprocessing_status="Standard torchvision ImageNet convention; downloaded checkpoint provenance unverified.",
        padding="Repeat last image to fill final batch; padded rows excluded from statistics.",
        model_provenance=a.model_provenance, threshold_rationale=a.threshold_rationale,
        limitation="External dataset evaluation. Training exclusion is not proven for an unknown checkpoint.",
        versions=dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__,
                      onnxruntime=ort.__version__))
    save_json(job / "manifest.json", manifest)
    print("PREPARED: %d images, %d OM batches. Copy this entire job to the NPU." %
          (len(samples), len(batches)))

def run_om(a):
    job = Path(a.job).resolve()
    m = read_json(job / "manifest.json")
    require(sha(job / "samples.json") == m["sample_manifest_sha256"], "Sample manifest changed")
    require(Path(a.om).is_file(), "OM model missing")
    exe = shutil.which(a.msame) or str(Path(a.msame).resolve())
    require(Path(exe).is_file(), "msame executable missing")
    # Never reuse an output directory: stale outputs must not be mistaken for new inference.
    run = fresh(a.run)
    om_digest = sha(a.om)
    meta = dict(config_id=a.config_id, om_path=str(Path(a.om).resolve()), om_sha256=om_digest,
                job_manifest_sha256=sha(job / "manifest.json"), device=a.device,
                batch_size=m["om_batch"], dynamic_batch=a.dynamic_batch,
                output_index=a.output_index, output_dtype_request=a.output_dtype,
                msame_sha256=sha(exe), host=platform.platform(),
                output_kind=m["output_kind"], status="running")
    save_json(run / "run_manifest.json", meta)
    all_outputs, records = [], []
    try:
        for b in m["batches"]:
            inp = inside(job, b["input_file"])
            require(sha(inp) == b["sha256"], "Input binary changed")
            out = run / ("batch_%05d" % b["index"])
            out.mkdir()
            cmd = [exe, "--model", str(Path(a.om).resolve()), "--input", str(inp),
                   "--output", str(out), "--outfmt", "BIN", "--loop", "1", "--device", str(a.device)]
            if a.dynamic_batch:
                cmd += ["--dymBatch", str(m["om_batch"])]
            tic = time.perf_counter()
            with (out / "msame.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=a.timeout, check=False)
            require(result.returncode == 0, "msame failed; inspect " + str(out / "msame.log"))
            candidates = [p for p in out.rglob("*.bin")
                          if re.search(r"output_?%d(?:_|\.|$)" % a.output_index, p.name, re.I)]
            if not candidates and a.output_index == 0:
                # Some versions use different names; only accept one unambiguous binary.
                candidates = list(out.rglob("*.bin"))
            require(len(candidates) == 1, "Expected one selected output binary; inspect " + str(out))
            p = candidates[0]
            total = m["om_batch"] * m["classes"]
            dtype = a.output_dtype
            if dtype == "auto":
                possibilities = [d for d in ("float16", "float32")
                                 if p.stat().st_size == total * np.dtype(d).itemsize]
                require(len(possibilities) == 1, "Output byte count does not match batch x classes")
                dtype = possibilities[0]
            require(p.stat().st_size == total * np.dtype(dtype).itemsize, "Wrong OM output dtype/shape")
            y = np.fromfile(p, dtype=np.dtype(dtype).newbyteorder("<")).reshape(m["om_batch"], m["classes"])
            y = classification_matrix(y, m["om_batch"], m["classes"])
            all_outputs.append(y[:b["valid"]].astype(np.float64))
            records.append(dict(batch_index=b["index"], valid=b["valid"], output=str(p.relative_to(run)),
                                output_sha256=sha(p), output_dtype=dtype,
                                invocation_seconds=time.perf_counter()-tic, command=cmd))
            print("OM batch %d/%d completed" % (b["index"]+1, len(m["batches"])), flush=True)
        require(sha(a.om) == om_digest, "OM changed during run")
        y = np.concatenate(all_outputs)
        require(len(y) == m["sample_count"], "OM output sample count mismatch")
        np.save(run / "om_outputs.npy", y, allow_pickle=False)
        meta.update(status="complete", rows=len(y), outputs_sha256=sha(run / "om_outputs.npy"),
                    batches=records,
                    timing_note="Per-batch model loading is used for isolation; not a latency benchmark.")
        save_json(run / "run_manifest.json", meta)
    except Exception as e:
        meta.update(status="failed", error=str(e), completed_batches=records)
        save_json(run / "run_manifest.json", meta)
        raise
    print("OM complete. Run compare on the NPU or copy the run directory back to the computer.")

def metrics(reference, deployed, epsilon=1e-12):
    require(reference.shape == deployed.shape and reference.ndim == 2, "Output shape mismatch")
    require(np.isfinite(reference).all() and np.isfinite(deployed).all(), "Non-finite outputs")
    r, d = reference.astype(np.float64), deployed.astype(np.float64)
    nr, nd = np.linalg.norm(r, axis=1), np.linalg.norm(d, axis=1)
    cos = np.full(len(r), np.nan)
    valid = (nr > epsilon) & (nd > epsilon)
    cos[valid] = np.clip(np.sum(r[valid]*d[valid], axis=1)/(nr[valid]*nd[valid]), -1, 1)
    err = np.linalg.norm(d-r, axis=1)/np.maximum(nr, epsilon)
    return r.argmax(1), d.argmax(1), cos, err

def describe(values):
    a = np.asarray(values)
    require(np.isfinite(a).all(), "Non-finite statistic")
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()), min=float(a.min()))

def compare(a):
    job, run = Path(a.job).resolve(), Path(a.run).resolve()
    m, rm = read_json(job / "manifest.json"), read_json(run / "run_manifest.json")
    require(rm["status"] == "complete", "OM run is incomplete/failed")
    require(sha(job / "manifest.json") == rm["job_manifest_sha256"], "Wrong job/run pairing")
    require(sha(job / "samples.json") == m["sample_manifest_sha256"], "Samples changed")
    require(sha(job / "reference_outputs.npy") == m["reference_sha256"], "Reference changed")
    require(sha(run / "om_outputs.npy") == rm["outputs_sha256"], "OM outputs changed")
    r = np.load(job / "reference_outputs.npy", allow_pickle=False)
    d = np.load(run / "om_outputs.npy", allow_pickle=False)
    samples = read_json(job / "samples.json")["samples"]
    require(r.shape == (len(samples), m["classes"]) and d.shape == r.shape, "Wrong output dimensions")
    if m["output_kind"] == "probabilities":
        for y in (r, d):
            require(np.all(y >= -1e-6) and np.allclose(y.sum(1), 1, atol=1e-3),
                    "Declared probabilities are not nonnegative/sum-to-one")
    rp, dp, cos, err = metrics(r, d, a.epsilon)
    agreement = rp == dp
    rows = []
    for i, sample in enumerate(samples):
        rows.append(dict(config_id=rm["config_id"], image_id=sample["image_id"],
            source_folder=sample["source_folder"], onnx_top1=int(rp[i]), om_top1=int(dp[i]),
            top1_agreement=int(agreement[i]),
            cosine_similarity=float(cos[i]) if np.isfinite(cos[i]) else "",
            cosine_defined=bool(np.isfinite(cos[i])),
            relative_l2_error=float(err[i]),
            max_absolute_error=float(np.max(np.abs(r[i]-d[i])))))
    report = fresh(a.report or str(run / "report"))
    csv_write(report / "per_image.csv", rows)
    finite_cos = cos[np.isfinite(cos)]
    hits, n = int(agreement.sum()), len(samples)
    summary = dict(config_id=rm["config_id"], dataset=m["dataset"], sample_count=n,
        matching_top1_count=hits, top1_agreement=hits/n, top1_agreement_percent=100*hits/n,
        relative_l2_error=describe(err), cosine=describe(finite_cos) if len(finite_cos) else None,
        undefined_cosine_count=int((~np.isfinite(cos)).sum()),
        cosine_threshold=a.cosine_threshold,
        cosine_threshold_pass_count=int(np.sum(cos >= a.cosine_threshold)),
        all_real_samples_pass_cosine_gate=bool(np.isfinite(cos).all() and np.all(cos >= a.cosine_threshold)),
        threshold_rationale=m["threshold_rationale"], epsilon=a.epsilon,
        onnx_reference_batch=m["reference_batch"], om_batch=m["om_batch"],
        output_kind=m["output_kind"], padded_rows_counted=False,
        task_accuracy_against_ground_truth=None,
        interpretation="Agreement with ONNX, not ground-truth accuracy. No new TOPSIS metric or admission threshold.",
        model_provenance=m["model_provenance"], held_out_status=m["limitation"],
        scope="Only this OM configuration and this fixed sample set; no all-81 claim.",
        reviewer_comment=REVIEW_COMMENT, software_version=VERSION)
    save_json(report / "summary.json", summary)
    save_json(report / "provenance.json", dict(job_manifest=m, run_manifest=rm))
    message = (
        "# Comment 6 validation result\n\n"
        "Configuration: %s\n\nImages: %d\n\nTop-1 agreement: %d/%d (%.4f%%)\n\n"
        "Relative L2 error: median %.8g; P95 %.8g; maximum %.8g.\n\n"
        "Undefined cosine values: %d. Padding rows are excluded.\n\n"
        "This is a deployment-consistency check, not classification accuracy or a new search score.\n"
        "Model training provenance is unverified unless independently documented.\n"
    ) % (rm["config_id"], n, hits, n, 100*hits/n, np.median(err),
         np.percentile(err,95), np.max(err), summary["undefined_cosine_count"])
    (report / "results.md").write_text(message, encoding="utf-8")
    print(message)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="Computer: sample images, create ONNX reference and OM input binaries")
    q.add_argument("--images", required=True)
    q.add_argument("--onnx", required=True)
    q.add_argument("--job", required=True)
    q.add_argument("--dataset-name", default="ImageNetV2 MatchedFrequency")
    q.add_argument("--samples", type=int, default=100)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--sampling", choices=["balanced","random"], default="balanced")
    q.add_argument("--sample-list", help="Reuse samples.json from a previous job")
    q.add_argument("--batch-size", type=int, default=4, help="Must match the actual OM input batch")
    q.add_argument("--ref-batch", type=int, help="Optional ONNX batch; fixed shape is otherwise detected")
    q.add_argument("--om-input-dtype", choices=["float32","float16"], default="float32")
    q.add_argument("--classes", type=int, default=1000)
    q.add_argument("--output-name")
    q.add_argument("--output-kind", choices=["logits","probabilities"], default="logits")
    q.add_argument("--model-provenance", default="Unknown downloaded checkpoint")
    q.add_argument("--threshold-rationale", default="Original 0.999 cutoff; selection rationale not documented")
    q.set_defaults(func=prepare)
    q = sub.add_parser("run-om", help="Ascend NPU: invoke existing msame; ONNX Runtime is not needed")
    q.add_argument("--job", required=True)
    q.add_argument("--om", required=True)
    q.add_argument("--msame", required=True)
    q.add_argument("--run", required=True)
    q.add_argument("--config-id", default="Exp67")
    q.add_argument("--device", type=int, default=0)
    q.add_argument("--dynamic-batch", action="store_true")
    q.add_argument("--output-index", type=int, default=0)
    q.add_argument("--output-dtype", choices=["auto","float32","float16"], default="auto")
    q.add_argument("--timeout", type=float, default=300)
    q.set_defaults(func=run_om)
    q = sub.add_parser("compare", help="Computer or NPU: pair outputs and report metrics")
    q.add_argument("--job", required=True)
    q.add_argument("--run", required=True)
    q.add_argument("--report")
    q.add_argument("--cosine-threshold", type=float, default=.999)
    q.add_argument("--epsilon", type=float, default=1e-12)
    q.set_defaults(func=compare)
    return p

if __name__ == "__main__":
    args = parser().parse_args()
    if hasattr(args, "epsilon"):
        require(args.epsilon > 0 and math.isfinite(args.epsilon), "epsilon must be positive/finite")
        require(-1 <= args.cosine_threshold <= 1, "cosine threshold outside [-1,1]")
    args.func(args)
