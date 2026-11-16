"""Latenza di ogni pipeline su 100 frame casuali di val, un frame alla volta, un processo per pipeline.

Uso: .venv/bin/python scripts/benchmark_latency.py            # tutte le pipeline, poi la tabella
     .venv/bin/python scripts/benchmark_latency.py --report   # solo la tabella dai json già salvati
"""
import argparse
import csv
import json
import multiprocessing
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJECT2 = REPO.parent / "kitticlassicfusion"               # L1, C1, FE, FM, FL-c, L2 girano dal loro codice
POWER = bool(os.environ.get("BENCH_POWER"))                # misura di potenza: 20 frame, finestre di tempo salvate
POWER_DIR = REPO / "results" / "latency" / "power"
OUT = POWER_DIR if POWER else REPO / "results" / "latency"
POWER_LOG = REPO / "results" / "latency" / "powermetrics.log"   # scritto da `sudo powermetrics`, vedi --power
POWER_FRAMES, IDLE_SECONDS = 20, 15
N_FRAMES, N_WARMUP, SEED = int(os.environ.get("BENCH_FRAMES", 100)), 3, 0
CLASSIC = ["L1", "C1", "FE", "FM", "FL-c", "L2"]
LEARNED = ["C2@mps", "C2@cpu", "C2s-extra", "fusion-extra"]   # pezzi misurati da soli e poi sommati
PARALLEL = ["FL-n@par-mps", "FL-n@par-cpu", "FL-c@par", "FM@par", "C2-s@par-mps"]   # i due rami davvero insieme
POWER_SET = ["L1", "C1", "FE", "FM@par", "FL-c@par", "L2", "C2@mps", "C2@cpu", "C2-s@par-mps", "FL-n@par-mps"]
PRETTY = {"FL-n@par-mps": "FL-n ‖ (L2 CPU, C2 mps)", "FL-n@par-cpu": "FL-n ‖ (L2 CPU, C2 cpu)", "FL-c@par": "FL-c ‖",
          "FM@par": "FM ‖", "C2-s@par-mps": "C2-s ‖ (stereo CPU, C2 mps)", "C2@mps": "C2 2D (mps)",
          "C2@cpu": "C2 2D (cpu)"}


# ================================================================ cronometri

class Clock:
    # tempo passato dentro le funzioni avvolte, diviso per nome di stadio
    def __init__(self):
        self.t = {}

    def wrap(self, fn, name):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
        return timed

    def take(self):
        t, self.t = self.t, {}
        return t

CLOCK = Clock()

class GpuEnergy:
    def __init__(self):
        import ctypes
        from ctypes import c_char_p, c_int32, c_int64, c_long, c_uint64, c_void_p
        self.ct = ctypes
        cf = self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        io = self.io = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        cf.CFStringCreateWithCString.restype, cf.CFStringCreateWithCString.argtypes = c_void_p, [c_void_p, c_char_p, c_int32]
        cf.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_long, c_int32]
        cf.CFDictionaryGetValue.restype, cf.CFDictionaryGetValue.argtypes = c_void_p, [c_void_p, c_void_p]
        cf.CFArrayGetCount.restype, cf.CFArrayGetCount.argtypes = c_long, [c_void_p]
        cf.CFArrayGetValueAtIndex.restype, cf.CFArrayGetValueAtIndex.argtypes = c_void_p, [c_void_p, c_long]
        io.IOReportCopyChannelsInGroup.restype = c_void_p
        io.IOReportCopyChannelsInGroup.argtypes = [c_void_p, c_void_p, c_uint64, c_uint64, c_uint64]
        io.IOReportCreateSubscription.restype = c_void_p
        io.IOReportCreateSubscription.argtypes = [c_void_p, c_void_p, ctypes.POINTER(c_void_p), c_uint64, c_void_p]
        io.IOReportCreateSamples.restype, io.IOReportCreateSamples.argtypes = c_void_p, [c_void_p, c_void_p, c_void_p]
        io.IOReportCreateSamplesDelta.restype = c_void_p
        io.IOReportCreateSamplesDelta.argtypes = [c_void_p, c_void_p, c_void_p]
        io.IOReportChannelGetChannelName.restype, io.IOReportChannelGetChannelName.argtypes = c_void_p, [c_void_p]
        io.IOReportSimpleGetIntegerValue.restype, io.IOReportSimpleGetIntegerValue.argtypes = c_int64, [c_void_p, c_int32]
        self.channels = c_void_p()
        self.sub = io.IOReportCreateSubscription(None, io.IOReportCopyChannelsInGroup(self.cfstr("Energy Model"), None, 0, 0, 0),
                                                 ctypes.byref(self.channels), 0, None)

    def cfstr(self, text):
        return self.cf.CFStringCreateWithCString(None, text.encode(), 0x08000100)

    def sample(self):
        return self.io.IOReportCreateSamples(self.sub, self.channels, None)

    def joules(self, before, after):
        items = self.cf.CFDictionaryGetValue(self.io.IOReportCreateSamplesDelta(before, after, None),
                                             self.cfstr("IOReportChannels"))
        buf = self.ct.create_string_buffer(128)
        for k in range(self.cf.CFArrayGetCount(items)):
            item = self.cf.CFArrayGetValueAtIndex(items, k)
            self.cf.CFStringGetCString(self.io.IOReportChannelGetChannelName(item), buf, 128, 0x08000100)
            if buf.value == b"GPU Energy":
                return self.io.IOReportSimpleGetIntegerValue(item, 0) * 1e-9   # nJ
        return float("nan")

WINDOW = {}

def time_io():
    # lettura e decodifica dei file: np.fromfile per il LiDAR, cv2.imread per le immagini
    import cv2
    import numpy as np
    np.fromfile = CLOCK.wrap(np.fromfile, "io")
    cv2.imread = CLOCK.wrap(cv2.imread, "io")

def sample_frames():
    from kitti.data import frame_ids
    frames = random.Random(SEED).sample(frame_ids("val"), N_FRAMES)
    return frames[:N_WARMUP], frames                        # il riscaldamento non entra nelle medie

def prewarm(frames):
    from config import DATA
    for i in frames:                                        # file già nella cache del sistema: io = decodifica, non disco
        for sub, ext in (("velodyne", "bin"), ("image_2", "png"), ("image_3", "png")):
            (DATA / sub / f"{i:06d}.{ext}").read_bytes()

def measure(run_frame, frames, warmup):
    for i in warmup:
        run_frame(i)
    CLOCK.take()
    rows = []
    gpu = GpuEnergy() if POWER else None
    if POWER:
        time.sleep(2)                                       # stacco netto dal riscaldamento nel log di potenza
        WINDOW.update(start=time.time(), gpu0=gpu.sample())
    for i in frames:
        c0, t0 = time.process_time(), time.perf_counter()
        extra = run_frame(i) or {}
        total, cpu = time.perf_counter() - t0, time.process_time() - c0
        stages = {**CLOCK.take(), **extra}
        if "total" in stages:                               # il pezzo ha cronometrato solo la sua parte
            total = stages.pop("total")
        info = stages.pop("info", {})                       # tempi del processo di fianco: non si sommano al totale
        rows.append({"frame": i, "total": total, "cpu": cpu + info.get("worker_cpu", 0.0), "stages": stages,
                     "info": info})
    if POWER:
        WINDOW.update(end=time.time(), gpu_j=gpu.joules(WINDOW.pop("gpu0"), gpu.sample()))
    return rows


# ================================================================ pipeline classiche (codice del progetto 2)

def classic_runner(name):
    sys.path.insert(0, str(PROJECT2 / "src"))
    time_io()
    from detectors import camera_classic, fusion_early, fusion_late, fusion_mid, lidar, lidar_net
    from kitti.data import load_calib
    camera_classic.compute_disparity = CLOCK.wrap(camera_classic.compute_disparity, "sgbm")
    if name == "L2":
        lidar_net.MODEL = lidar_net.load_model()            # senza il set_num_threads(1) pensato per run.py
        lidar_net.proposals = CLOCK.wrap(lidar_net.proposals, "proposals")
        lidar_net.predict = CLOCK.wrap(lidar_net.predict, "pointnet")
    detect = {"L1": lidar.detect, "C1": camera_classic.detect, "FE": fusion_early.detect,
              "FM": fusion_mid.detect, "FL-c": fusion_late.detect, "L2": lidar_net.detect}[name]
    calibs = {i: load_calib(i) for i in sample_frames()[1]}   # la calibrazione è fissa: fuori dal tempo

    def run_frame(i):
        calib = calibs[i]
        t0 = time.perf_counter()
        detect(i, calib)
        return {"total": time.perf_counter() - t0}
    return run_frame


# ================================================================ pezzi delle pipeline di questo progetto

def learned_runner(name):
    sys.path[:0] = [str(REPO / "src"), str(REPO / "src" / "methods")]
    time_io()
    import numpy as np
    import torch
    from kitti.data import load_calib, load_image

    if name.startswith("C2@"):
        from config import CLASSES
        from fine_tuning import WEIGHTS, load_model
        device = torch.device(name.split("@")[1])
        model = load_model()
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        model.to(device).eval()

        def run_frame(i):
            t0 = time.perf_counter()
            img = load_image(i)
            t1 = time.perf_counter()
            x = (torch.from_numpy(img).permute(2, 0, 1).float() / 255).to(device)
            t2 = time.perf_counter()
            with torch.inference_mode():
                out = model([x])[0]
                if device.type == "mps":
                    torch.mps.synchronize()                 # altrimenti misuro solo il lancio
            t3 = time.perf_counter()
            boxes, scores = out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()
            [CLASSES[k - 1] for k in out["labels"].tolist()]
            return {"total": time.perf_counter() - t0, "prep": t2 - t1, "inference": t3 - t2,
                    "post": time.perf_counter() - t3}
        return run_frame

    if name == "C2s-extra":                                 # oltre a C2: immagine destra, stereo, box 3D
        import camera_stereo as cs
        params = json.loads(cs.PARAMS_PATH.read_text())
        cs.compute_disparity = CLOCK.wrap(cs.compute_disparity, "sgbm")
        inputs = {}
        for i in sample_frames()[1]:                        # box di C2 dalla cache e immagine sinistra: già contati in C2
            z = np.load(cs.C2_CACHE / f"{i:06d}.npz")
            inputs[i] = (load_calib(i), load_image(i), (z["boxes"], z["labels"], z["scores"]))

        def run_frame(i):
            calib, left, det2d = inputs[i]
            t0 = time.perf_counter()
            right = load_image(i, 3)
            lifted = cs.lift_boxes(det2d, left, right, calib)
            boxes = cs.make_boxes(lifted, params)
            [k for k, c in enumerate(lifted["labels"]) if lifted["scores"][k] >= params["threshold"][c]]
            return {"total": time.perf_counter() - t0}
        return run_frame

    if name == "fusion-extra":                              # oltre a L2 e C2: proiezione, coppie, MLP, soglia
        import clocs
        saved = torch.load(clocs.MODEL_PATH, map_location="cpu")
        model = clocs.Clocs(clocs.n_inputs(saved.get("features", [])))
        model.load_state_dict(saved["weights"])
        model.eval()
        cands, cand_classes = clocs.load_candidates("val")
        inputs = {i: (load_calib(i), clocs.image_size(i), clocs.load_det2d(i, saved["det2d"]))
                  for i in sample_frames()[1]}              # candidati di L2 e box di C2 dalla cache: contati altrove

        def run_frame(i):
            calib, size, det2d = inputs[i]
            t0 = time.perf_counter()
            cand = clocs.decode_candidates(cands[i], cand_classes)
            if len(cand["boxes"]):
                scores = clocs.fused_scores(model, clocs.frame_pairs(cand, calib, size, det2d, saved.get("features", [])))
                [k for k in range(len(scores)) if scores[k] >= saved["thresholds"][cand["classes"][k]]]
            return {"total": time.perf_counter() - t0}
        return run_frame
    raise SystemExit(f"pipeline sconosciuta: {name}")


# ================================================================ fusioni con i due rami in parallelo

def worker_main(conn, kind, frames):
    # processo di fianco, con il codice del progetto 2: riceve un frame, calcola il suo ramo, lo rimanda
    sys.path.insert(0, str(PROJECT2 / "src"))
    import numpy as np
    from detectors import camera_classic, lidar, lidar_net
    from kitti.data import load_calib
    from networks.pointnet import decode_box
    calibs = {i: load_calib(i) for i in frames}
    if kind == "l2cand":
        lidar_net.MODEL = lidar_net.load_model()

    def l2_candidates(i):
        # come L2 ma senza soglia: tutti i candidati con box corretti, probabilità e numero di punti
        boxes, clusters, _, _ = lidar_net.proposals(lidar_net.cloud(i), calibs[i])
        if len(boxes) == 0:
            return {"boxes": np.zeros((0, 7)), "probs": np.zeros((0, 4), dtype=np.float32), "n_points": np.zeros(0, int)}
        probs, reg = lidar_net.predict(boxes, clusters, lidar_net.MODEL[0])
        return {"boxes": np.array([decode_box(b, r) for b, r in zip(boxes, reg)]), "probs": probs,
                "n_points": np.array([len(c) for c in clusters])}

    fn = {"l2cand": l2_candidates, "l1": lambda i: lidar.detect(i, calibs[i]),
          "stereo_cloud": lambda i: camera_classic.stereo_cloud(i, calibs[i])}[kind]
    conn.send("ready")
    while True:
        i = conn.recv()
        if i is None:
            conn.send(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
            return
        c0, t0 = time.process_time(), time.perf_counter()
        out = fn(i)
        conn.send((out, time.perf_counter() - t0, time.process_time() - c0))

class Worker:
    def __init__(self, kind, frames):
        self.conn, child = multiprocessing.get_context("spawn").Pipe()
        self.proc = multiprocessing.get_context("spawn").Process(target=worker_main, args=(child, kind, frames))
        self.proc.start()
        self.conn.recv()                                    # pronto: modello caricato

    def start(self, i):
        self.conn.send(i)

    def result(self):
        return self.conn.recv()

    def close(self):
        self.conn.send(None)
        rss = self.conn.recv()
        self.proc.join()
        return rss

WORKER = None

def parallel_runner(name):
    global WORKER
    frames = None
    if name.startswith("FL-n@par"):                         # L2 sulla CPU nel processo di fianco, C2 qui (GPU o CPU)
        sys.path[:0] = [str(REPO / "src"), str(REPO / "src" / "methods")]
        import numpy as np
        import torch
        import clocs
        from config import CLASSES
        from fine_tuning import WEIGHTS, load_model
        from kitti.data import load_calib, load_image
        frames = sample_frames()[1]
        device = torch.device(name.split("-")[-1])
        c2 = load_model()
        c2.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        c2.to(device).eval()
        saved = torch.load(clocs.MODEL_PATH, map_location="cpu")
        features = saved.get("features", [])
        net = clocs.Clocs(clocs.n_inputs(features))
        net.load_state_dict(saved["weights"])
        net.eval()
        cand_classes = clocs.load_candidates("val")[1]
        calibs = {i: load_calib(i) for i in frames}
        WORKER = Worker("l2cand", frames)

        def run_frame(i):
            t0 = time.perf_counter()
            WORKER.start(i)                                 # il ramo LiDAR parte subito
            img = load_image(i)
            x = (torch.from_numpy(img).permute(2, 0, 1).float() / 255).to(device)
            with torch.inference_mode():
                out = c2([x])[0]
            det2d = (out["boxes"].cpu().numpy(), np.array([CLASSES[k - 1] for k in out["labels"].tolist()]),
                     out["scores"].cpu().numpy())           # .cpu() aspetta la fine dei calcoli sulla GPU
            t1 = time.perf_counter()
            raw, l2_wall, l2_cpu = WORKER.result()
            t2 = time.perf_counter()
            cand = clocs.decode_candidates(raw, cand_classes)
            n = 0
            if len(cand["boxes"]):
                size = (img.shape[1], img.shape[0])
                scores = clocs.fused_scores(net, clocs.frame_pairs(cand, calibs[i], size, det2d, features))
                n = sum(scores[k] >= saved["thresholds"][cand["classes"][k]] for k in range(len(scores)))
            t3 = time.perf_counter()
            return {"total": t3 - t0, "ramo C2": t1 - t0, "attesa di L2": t2 - t1, "fusione": t3 - t2,
                    "info": {"worker_wall": l2_wall, "worker_cpu": l2_cpu, "n_det": int(n)}}
        return run_frame

    if name.startswith("C2-s@par"):                         # stesso processo: OpenCV rilascia il GIL durante SGBM
        sys.path[:0] = [str(REPO / "src"), str(REPO / "src" / "methods")]
        import threading
        import numpy as np
        import torch
        import camera_stereo as cs
        from config import CLASSES
        from fine_tuning import WEIGHTS, load_model
        from kitti.data import load_calib, load_image
        from stereo.pseudo_lidar import disparity_to_depth, remove_flying_pixels, stereo_params
        frames = sample_frames()[1]
        device = torch.device(name.split("-")[-1])
        c2 = load_model()
        c2.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        c2.to(device).eval()
        params = json.loads(cs.PARAMS_PATH.read_text())
        calibs = {i: load_calib(i) for i in frames}

        def run_frame(i):
            t0 = time.perf_counter()
            left = load_image(i)
            box = {}

            def stereo():                                   # ramo stereo: immagine destra, SGBM, profondità
                s0 = time.perf_counter()
                f, _, _, baseline = stereo_params(calibs[i])
                disp = cs.compute_disparity(left, load_image(i, 3))
                box["depth"] = remove_flying_pixels(disparity_to_depth(disp, f, baseline), cs.MAX_JUMP)
                box["wall"] = time.perf_counter() - s0
            th = threading.Thread(target=stereo)
            th.start()
            x = (torch.from_numpy(left).permute(2, 0, 1).float() / 255).to(device)
            with torch.inference_mode():
                out = c2([x])[0]
            det = (out["boxes"].cpu().numpy(), [CLASSES[k - 1] for k in out["labels"].tolist()], out["scores"].cpu().numpy())
            t1 = time.perf_counter()
            th.join()
            t2 = time.perf_counter()
            lifted = cs.lift_boxes(det, None, None, calibs[i], depth=box["depth"])
            boxes = cs.make_boxes(lifted, params)
            [k for k, c in enumerate(lifted["labels"]) if lifted["scores"][k] >= params["threshold"][c]]
            t3 = time.perf_counter()
            CLOCK.take()
            return {"total": t3 - t0, "ramo C2": t1 - t0, "attesa dello stereo": t2 - t1, "box 3D": t3 - t2,
                    "info": {"worker_wall": box["wall"]}}
        return run_frame

    sys.path.insert(0, str(PROJECT2 / "src"))               # FL-c e FM: tutti e due i rami sono codice del progetto 2
    import numpy as np
    from detectors import camera_classic, fusion_mid
    from fusion.late import confirmed
    from fusion.occupancy import fused_boxes
    from kitti.data import load_calib
    from pipeline.classic import in_camera_fov, rule_detections
    from pipeline.floor import ransac_floor
    frames = sample_frames()[1]
    calibs = {i: load_calib(i) for i in frames}

    if name == "FL-c@par":                                  # L1 nel processo di fianco, C1 qui: due core della CPU
        WORKER = Worker("l1", frames)

        def run_frame(i):
            t0 = time.perf_counter()
            WORKER.start(i)
            cam = camera_classic.detect(i, calibs[i])
            t1 = time.perf_counter()
            lid, wall, cpu = WORKER.result()
            t2 = time.perf_counter()
            confirmed(lid, cam)
            t3 = time.perf_counter()
            return {"total": t3 - t0, "ramo C1": t1 - t0, "attesa di L1": t2 - t1, "fusione": t3 - t2,
                    "info": {"worker_wall": wall, "worker_cpu": cpu}}
        return run_frame

    if name == "FM@par":                                    # nuvola stereo di fianco, suolo dal LiDAR qui
        WORKER = Worker("stereo_cloud", frames)

        def run_frame(i):
            t0 = time.perf_counter()
            WORKER.start(i)
            lidar = fusion_mid.cloud(i)
            _, n, d = ransac_floor(lidar)
            t1 = time.perf_counter()
            stereo, wall, cpu = WORKER.result()
            t2 = time.perf_counter()
            boxes = fused_boxes(lidar, stereo, n, d)
            rule_detections(np.array([b for b in boxes if in_camera_fov(b, calibs[i])]).reshape(-1, 7))
            t3 = time.perf_counter()
            return {"total": t3 - t0, "ramo LiDAR (suolo)": t1 - t0, "attesa dello stereo": t2 - t1,
                    "griglia fusa e box": t3 - t2, "info": {"worker_wall": wall, "worker_cpu": cpu}}
        return run_frame
    raise SystemExit(f"pipeline sconosciuta: {name}")


def run_one(name):
    if name in PARALLEL:
        run_frame = parallel_runner(name)
    else:
        run_frame = classic_runner(name) if name in CLASSIC else learned_runner(name)
    warmup, frames = sample_frames()
    prewarm(frames)
    rows = measure(run_frame, frames, warmup)
    info = {"name": name, "rows": rows, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    if WORKER is not None:
        info["worker_rss_mb"] = WORKER.close()
    info["window"] = WINDOW
    if name in ("C2@mps", "FL-n@par-mps"):
        import torch
        info["mps_driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(info))


# ================================================================ tabella

def load(name):
    return json.loads((OUT / f"{name}.json").read_text())

def per_frame(name):
    # frame -> (totale, cpu, stadi) di un pezzo misurato
    return {r["frame"]: r for r in load(name)["rows"]}

def compose(parts, device="mps"):
    # somma, frame per frame, i pezzi di una pipeline; gli stadi prendono il nome del pezzo
    data = [(p.split("@")[0], per_frame(p)) for p in parts]
    rows = []
    for i in data[0][1]:
        stages = {}
        for label, d in data:
            r = d[i]
            rest = r["total"] - sum(r["stages"].values())
            for k, v in r["stages"].items():
                stages[f"{label}: {k}"] = v
            stages[f"{label}: resto"] = rest
        rows.append({"frame": i, "total": sum(d[i]["total"] for _, d in data),
                     "cpu": sum(d[i]["cpu"] for _, d in data), "stages": stages})
    return rows

def stats(rows):
    import numpy as np
    t = np.array([r["total"] for r in rows]) * 1000
    cpu = np.array([r["cpu"] for r in rows]) * 1000
    io = np.array([sum(v for k, v in r["stages"].items() if k.split(": ")[-1] == "io") for r in rows]) * 1000
    return {"mean": t.mean(), "std": t.std(), "median": np.median(t), "p95": np.percentile(t, 95), "max": t.max(),
            "fps": 1000 / t.mean(), "io": io.mean(), "cores": cpu.sum() / t.sum()}

def stage_means(rows):
    import numpy as np
    keys = list(dict.fromkeys(k for r in rows for k in r["stages"]))
    return {k: float(np.mean([r["stages"].get(k, 0.0) for r in rows]) * 1000) for k in keys}

def val_f1():
    f1 = {}
    with open(REPO / "results" / "val" / "comparison.csv") as f:
        for r in csv.DictReader(f):
            f1.setdefault(r["method"], {})[r["class"]] = float(r["F1"])
    return f1

def report():
    def single(name):
        rows = load(name)["rows"]
        for r in rows:
            r["stages"]["resto"] = r["total"] - sum(r["stages"].values())
        return rows

    pipelines = {m: single(m) for m in CLASSIC}
    for dev in ("mps", "cpu"):
        pipelines[f"C2 2D ({dev})"] = compose([f"C2@{dev}"])
        pipelines[f"C2-s ({dev})"] = compose([f"C2@{dev}", "C2s-extra"])
        pipelines[f"FL-n ({dev})"] = compose(["L2", f"C2@{dev}", "fusion-extra"])
    par = [m for m in PARALLEL if (OUT / f"{m}.json").exists()]
    for m in par:                                           # misurate intere: i due rami girano insieme
        pipelines[PRETTY[m]] = single(m)
    mem = {m: load(m)["peak_rss_mb"] for m in CLASSIC + LEARNED}
    mem.update({m: load(m)["peak_rss_mb"] + load(m).get("worker_rss_mb", 0) for m in par})
    branch = {m: sum(r["info"]["worker_wall"] for r in load(m)["rows"]) / N_FRAMES * 1000 for m in par}
    f1 = val_f1()

    lines = [f"Latenza per frame su {N_FRAMES} frame casuali di val (seed {SEED}), un frame alla volta, in ms", "",
             "| pipeline | media | dev. std | mediana | p95 | max | fps | di cui lettura file | core usati |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    st = {}
    for name, rows in pipelines.items():
        s = st[name] = stats(rows)
        lines.append(f"| {name} | {s['mean']:.0f} | {s['std']:.0f} | {s['median']:.0f} | {s['p95']:.0f} | "
                     f"{s['max']:.0f} | {s['fps']:.2f} | {s['io']:.0f} | {s['cores']:.1f} |")
    lines += ["", "Stadi (media in ms)", ""]
    for name, rows in pipelines.items():
        parts = ", ".join(f"{k} {v:.0f}" for k, v in stage_means(rows).items() if v >= 0.5)
        lines.append(f"- **{name}**: {parts}")
    lines += ["", "Memoria di picco del processo (MB)", "",
              ", ".join(f"{m} {v:.0f}" for m, v in mem.items()),
              f"C2@mps: {load('C2@mps').get('mps_driver_mb', 0):.0f} MB allocati dal driver della GPU", "",
              "Ramo nel processo di fianco, tempo proprio mentre gira insieme all'altro (ms): "
              + ", ".join(f"{m} {v:.0f}" for m, v in branch.items()), "",
              "Costo contro guadagno (F1 3D su val)", "",
              "| confronto | ms in più | tempo × | Δ F1 Car | Δ F1 Pedestrian | Δ F1 Cyclist |", "|---|---:|---:|---:|---:|---:|"]
    pairs = [("FL-n (mps)", "L2", "FL-n", "L2"), ("FL-n (cpu)", "L2", "FL-n", "L2"),
             ("FL-n ‖ (L2 CPU, C2 mps)", "L2", "FL-n", "L2"), ("FL-n ‖ (L2 CPU, C2 cpu)", "L2", "FL-n", "L2"),
             ("L2", "L1", "L2", "L1"), ("FL-c", "L1", "FL-c", "L1"), ("FL-c ‖", "L1", "FL-c", "L1"),
             ("FE", "L1", "FE", "L1"), ("FM", "L1", "FM", "L1"), ("FM ‖", "L1", "FM", "L1"),
             ("C2-s (mps)", "C1", "C2-s", "C1"), ("C2-s ‖ (stereo CPU, C2 mps)", "C1", "C2-s", "C1")]
    for a, b, fa, fb in [x for x in pairs if x[0] in st]:
        d = [f1[fa][c] - f1[fb][c] for c in ("Car", "Pedestrian", "Cyclist")]
        lines.append(f"| {a} vs {b} | {st[a]['mean'] - st[b]['mean']:+.0f} | {st[a]['mean'] / st[b]['mean']:.2f} | "
                     + " | ".join(f"{x:+.3f}" for x in d) + " |")
    (OUT / "latency.md").write_text("\n".join(lines) + "\n")
    (OUT / "latency.json").write_text(json.dumps({n: {**st[n], "stages": stage_means(r)} for n, r in pipelines.items()},
                                                 indent=1))
    print("\n".join(lines))


# ================================================================ potenza

def read_powermetrics():
    # campioni di `powermetrics --samplers cpu_power,gpu_power`: (inizio, fine, {"CPU": W, "GPU": W, ...})
    import re
    from datetime import datetime
    if not POWER_LOG.exists():
        return []
    blocks = re.split(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\) \*\*\*", POWER_LOG.read_text(errors="ignore"))
    stamps, spans, values = [], [], []
    for when, ms, body in zip(blocks[1::3], blocks[2::3], blocks[3::3]):
        power = {}
        for name, mw in re.findall(r"^(CPU|GPU|ANE|Combined) Power[^:]*: (\d+) mW", body, flags=re.M):
            power.setdefault(name, int(mw) / 1000)          # la GPU compare due volte: tengo la prima
        if "CPU" in power:
            stamps.append(datetime.strptime(when, "%a %b %d %H:%M:%S %Y %z").timestamp())
            spans.append(float(ms) / 1000)
            values.append(power)
    if not stamps:
        return []
    ends, t = [], 0.0
    for d in spans:                                         # tempo esatto dai ms trascorsi: le date hanno solo i secondi
        t += d
        ends.append(t)
    offset = sum(a - b for a, b in zip(stamps, ends)) / len(stamps) + 0.5
    return [(e + offset - d, e + offset, v) for e, d, v in zip(ends, spans, values)]

def window_power(samples, start, end, margin=0.3):
    inside = [(b - a, v) for a, b, v in samples if a >= start + margin and b <= end - margin]
    total = sum(d for d, _ in inside)
    if not inside:
        return None
    return {k: sum(d * v.get(k, 0.0) for d, v in inside) / total for k in ("CPU", "GPU", "ANE", "Combined")}

def power_run():
    recent = POWER_LOG.exists() and time.time() - POWER_LOG.stat().st_mtime < 5
    if not recent:
        print(f"ATTENZIONE: {POWER_LOG} non sta crescendo: senza `sudo powermetrics` resta solo l'energia della GPU.\n"
              f"  sudo powermetrics --samplers cpu_power,gpu_power -i 200 -o {POWER_LOG}", flush=True)
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    gpu = GpuEnergy()
    time.sleep(3)
    window = {"start": time.time(), "g": gpu.sample()}
    time.sleep(IDLE_SECONDS)                                # riferimento: macchina ferma
    (POWER_DIR / "idle.json").write_text(json.dumps({"window": {"start": window["start"], "end": time.time(),
                                                                "gpu_j": gpu.joules(window["g"], gpu.sample())}}))
    env = {**os.environ, "BENCH_POWER": "1", "BENCH_FRAMES": str(POWER_FRAMES)}
    for name in POWER_SET:
        time.sleep(3)                                       # pausa: le finestre non si toccano
        t0 = time.time()
        subprocess.run([sys.executable, __file__, "--pipeline", name], check=True, env=env)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    time.sleep(2)                                           # gli ultimi campioni arrivano nel file

def power_report():
    samples = read_powermetrics()
    f1 = val_f1()
    idle_info = json.loads((POWER_DIR / "idle.json").read_text())["window"]
    idle = window_power(samples, idle_info["start"], idle_info["end"])
    idle_gpu = idle_info["gpu_j"] / (idle_info["end"] - idle_info["start"])
    lines = [f"Potenza media durante {POWER_FRAMES} frame per pipeline (stessi primi frame della latenza), in W; "
             "energia per frame in J", ""]
    if idle:
        lines += [f"Macchina ferma ({IDLE_SECONDS} s): CPU {idle['CPU']:.2f} W, GPU {idle['GPU']:.2f} W, "
                  f"totale {idle['Combined']:.2f} W", ""]
    else:
        lines += [f"Nessun campione di powermetrics: solo la GPU (contatore IOReport). GPU ferma: {idle_gpu:.2f} W", ""]
    lines += ["| pipeline | ms/frame | CPU W | GPU W | CPU+GPU W | J/frame | J/frame sopra il fermo | GPU W (IOReport) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    table = {}
    for name in POWER_SET:
        info = json.loads((POWER_DIR / f"{name}.json").read_text())
        w, n = info["window"], len(info["rows"])
        dur = w["end"] - w["start"]
        ms = sum(r["total"] for r in info["rows"]) / n * 1000
        pw = window_power(samples, w["start"], w["end"])
        row = {"ms": ms, "gpu_io": w["gpu_j"] / dur}
        if pw:
            row.update(cpu=pw["CPU"], gpu=pw["GPU"], tot=pw["Combined"], j=pw["Combined"] * dur / n,
                       j_net=(pw["Combined"] - idle["Combined"]) * dur / n)
        table[name] = row
        f = (lambda k, d=2: f"{row[k]:.{d}f}" if k in row else "n/d")
        lines.append(f"| {PRETTY.get(name, name)} | {ms:.0f} | {f('cpu')} | {f('gpu')} | {f('tot')} | {f('j')} | "
                     f"{f('j_net')} | {row['gpu_io']:.2f} |")
    if all("j" in r for r in table.values()):
        lines += ["", "Energia contro guadagno (F1 3D su val)", "",
                  "| confronto | J/frame in più | energia × | Δ F1 Car | Δ F1 Pedestrian | Δ F1 Cyclist |",
                  "|---|---:|---:|---:|---:|---:|"]
        for a, b, fa, fb in [("FL-n@par-mps", "L2", "FL-n", "L2"), ("L2", "L1", "L2", "L1"),
                             ("FL-c@par", "L1", "FL-c", "L1"), ("FE", "L1", "FE", "L1"), ("FM@par", "L1", "FM", "L1"),
                             ("C2-s@par-mps", "C1", "C2-s", "C1")]:
            d = [f1[fa][c] - f1[fb][c] for c in ("Car", "Pedestrian", "Cyclist")]
            lines.append(f"| {PRETTY.get(a, a)} vs {b} | {table[a]['j'] - table[b]['j']:+.2f} | "
                         f"{table[a]['j'] / table[b]['j']:.2f} | " + " | ".join(f"{x:+.3f}" for x in d) + " |")
    (POWER_DIR / "power.md").write_text("\n".join(lines) + "\n")
    (POWER_DIR / "power.json").write_text(json.dumps({"idle": idle, "pipelines": table}, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", help="misura una sola pipeline in questo processo")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", help="rimisura solo questa pipeline (o 'parallel'), poi rifà la tabella")
    ap.add_argument("--power", action="store_true",
                    help=f"potenza su {POWER_FRAMES} frame; prima, in un altro terminale: "
                         f"sudo powermetrics --samplers cpu_power,gpu_power -i 200 -o {POWER_LOG}")
    ap.add_argument("--power-report", action="store_true", help="rifà solo la tabella di potenza")
    args = ap.parse_args()
    if args.pipeline:
        return run_one(args.pipeline)
    if args.power or args.power_report:
        if args.power:
            power_run()
        return power_report()
    if not args.report:
        todo = PARALLEL if args.only == "parallel" else [args.only] if args.only else CLASSIC + LEARNED + PARALLEL
        for name in todo:                                   # una alla volta: niente contesa fra pipeline
            t0 = time.time()
            subprocess.run([sys.executable, __file__, "--pipeline", name], check=True)
            print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    sys.path.insert(0, str(REPO / "src"))
    report()


if __name__ == "__main__":
    main()
