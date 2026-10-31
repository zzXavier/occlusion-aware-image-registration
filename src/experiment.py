"""Occluded 2D registration. Flow convention is target-to-source (dx, dy), in pixels."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import skimage
import torch
from PIL import Image, ImageDraw
from voxelmorph.nn.models import VxmPairwise

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/experiment.json").read_text())
DATA = Path(skimage.__file__).parent / "data"
METHODS = ("identity", "translation", "farneback", "vxm", "vxm_masked")


def pair(name, seed):
    rng = np.random.default_rng(seed)
    with Image.open(DATA / name) as photo:
        gray = photo.convert("L")
        crop = int(min(gray.size) * rng.uniform(.5, .95))
        left, top = (rng.integers(0, max(1, length - crop + 1)) for length in gray.size)
        n = CFG["size"]
        img = np.array(gray.crop((left, top, left + crop, top + crop)).resize((n, n), Image.Resampling.LANCZOS))
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, 2)
    dx = rng.uniform(-3, 3) + 2.5 * np.sin(y * 2 * np.pi / n + phase[0])
    dy = rng.uniform(-3, 3) + 2.5 * np.cos(x * 2 * np.pi / n + phase[1])
    truth = np.stack((dx, dy), axis=-1).astype(np.float32)
    coords = np.stack((x, y), axis=-1) + truth
    fixed = cv2.remap(img, coords, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    visible = np.ones_like(img, dtype=bool)
    h, w = rng.integers(n//5, n//2, 2)
    i, j = rng.integers(3, n-max(h,w)-2, 2)
    visible[i:i+h, j:j+w] = False
    valid = ((coords >= 1) & (coords < n-1)).all(axis=-1)
    fixed[~visible] = 0
    return img, fixed, truth, visible, valid


def model():
    return VxmPairwise(2, 1, 1, nb_features=[8, 12, 16], integration_steps=0)


def tensors(items, idx):
    return torch.from_numpy(np.stack([item[idx] for item in items])).unsqueeze(1).float()


def fit(masked, out):
    torch.manual_seed(CFG["seed"])
    network = model()
    optimizer = torch.optim.Adam(network.parameters(), lr=.002)
    rng = np.random.default_rng(CFG["seed"])
    train_log = []
    for step in range(CFG["train_steps"]):
        batch = [pair(rng.choice(CFG["train"]), int(rng.integers(2**31))) for _ in range(CFG["batch"])]
        moving, fixed = tensors(batch, 0)/255, tensors(batch, 1)/255
        flow, warped = network(moving, fixed, return_warped_source=True)
        weight = tensors(batch, 3) if masked else torch.ones_like(fixed)
        photo = (weight * (fixed - warped).square()).sum() / weight.sum()
        smooth = flow.diff(dim=2).square().mean() + flow.diff(dim=3).square().mean()
        loss = photo + .015 * smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.append(float(loss.detach()))
    path = out / ("masked.pt" if masked else "unmasked.pt")
    torch.save(network.state_dict(), path)
    loaded = model()
    loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(network(moving, fixed), loaded(moving, fixed), rtol=0, atol=0)
    return loaded.eval(), train_log


def geom_baselines(moving, fixed):
    zero = np.zeros((*moving.shape, 2), dtype=np.float32)
    shift, _ = cv2.phaseCorrelate(fixed.astype("f"), moving.astype("f"))
    translated = np.full_like(zero, shift)
    dense = cv2.calcOpticalFlowFarneback(fixed, moving, None, .5, 3, 17, 4, 5, 1.2, 0)
    return [zero, translated, dense]


def metrics(flow, truth, visible, valid):
    epe = np.linalg.norm(flow - truth, axis=-1)
    y, x = np.indices(visible.shape)
    landmarks = (x % 4 == 2) & (y % 4 == 2) & valid
    u, v = flow[...,0], flow[...,1]
    uy, ux = np.gradient(u)
    vy, vx = np.gradient(v)
    jac = (1+ux)*(1+vy) - uy*vx
    return [float(epe[landmarks].mean()), float(epe[valid & visible].mean()),
            float(epe[valid & ~visible].mean()), float(np.mean(jac[valid] <= 0))]


def figure(sample, flow, out):
    moving, fixed, truth, _, _ = sample
    error = np.linalg.norm(flow-truth, axis=-1)
    heat = Image.fromarray((np.clip(error/8, 0, 1)*255).astype("uint8"))
    width = 192
    canvas = Image.new("RGB", (width*3, width+24), "white")
    labels = ["Moving", "Target + occlusion", "Masked VXM error 0-8px"]
    for i, img in enumerate([Image.fromarray(moving), Image.fromarray(fixed), heat]):
        canvas.paste(img.resize((width,width)), (width*i,24))
        ImageDraw.Draw(canvas).text((4+width*i,4), labels[i], fill="black")
    canvas.save(out)


def reproduce(out):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    out.mkdir(parents=True, exist_ok=True)
    unmasked, log0 = fit(False, out)
    masked, log1 = fit(True, out)
    logs = dict(unmasked=log0, masked=log1)
    (out/"training.json").write_text(json.dumps(logs))
    records = []
    for name in CFG["test"]:
        for idx in range(CFG["holdout_per_image"]):
            pair_seed = CFG["seed"] + 10000 + idx
            sample = pair(name, pair_seed)
            moving, fixed, truth, visible, valid = sample
            flows = geom_baselines(moving, fixed)
            with torch.inference_mode():
                for trained in (unmasked, masked):
                    f = trained(tensors([sample], 0)/255, tensors([sample], 1)/255)
                    flows.append(f[0].numpy()[[1,0]].transpose(1,2,0))
            records.append((flows, truth, visible, valid))
            if name == CFG["test"][0] and idx == 0:
                figure(sample, flows[-1], out/"comparison.png")
    flows, truth, visible, valid = map(np.stack, zip(*records))
    np.savez_compressed(out/"predictions.npz", flows=flows, truth=truth, visible=visible, valid=valid)
    manifest = {k:hashlib.sha256((DATA/k).read_bytes()).hexdigest() for k in CFG["train"]+CFG["test"]}
    (out/"data-sha256.json").write_text(json.dumps(manifest, indent=2))
    evaluate(out)


def evaluate(out):
    with np.load(out/"predictions.npz") as d:
        allrows = []
        for i, flows in enumerate(d["flows"]):
            name = CFG["test"][i//CFG["holdout_per_image"]]
            for method, flow in zip(METHODS, flows):
                allrows.append([name, i % CFG["holdout_per_image"], method,
                    *metrics(flow, d["truth"][i], d["visible"][i], d["valid"][i])])
    header = ["image","case","method","landmark_tre_px","visible_epe_px","occluded_epe_px","fold_fraction"]
    with (out/"cases.csv").open("w",newline="") as fp:
        writer=csv.writer(fp); writer.writerow(header); writer.writerows(allrows)
    with (out/"summary.csv").open("w",newline="") as fp:
        writer=csv.writer(fp); writer.writerow(header[2:])
        for method in METHODS:
            rows = [r[3:] for r in allrows if r[2]==method]
            writer.writerow([method,*np.mean(rows,axis=0)])
    print((out/"summary.csv").read_text())


def demo(out):
    torch.set_num_threads(2)
    sample = pair(CFG["test"][0], CFG["seed"]+10000)
    net = model().eval()
    net.load_state_dict(torch.load(ROOT/"results/masked.pt", weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        flow = net(tensors([sample],0)/255, tensors([sample],1)/255)[0].numpy()[[1,0]].transpose(1,2,0)
    out.mkdir(parents=True, exist_ok=True)
    figure(sample, flow, out/"demo.png")
    print(out/"demo.png")
