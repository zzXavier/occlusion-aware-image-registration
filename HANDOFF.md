# Occlusion-aware image registration handoff

## Latest verified state · 2026-11-03

The public README and experiment artifacts are available from the [repository](https://github.com/zzXavier/occlusion-aware-image-registration).

Verified experiment revision: `e5c332447ec5f8464ba5639ab663516a2e569da0`. A new GitHub checkout trained both VoxelMorph arms, reloaded each checkpoint, ran the demo and reproduced the scored table for all 18 held-out pairs.

Next local command: `python run.py demo --out scratch/demo`.

## Earlier implementation notes


Local run `cpu-1769` completed on Windows, Python 3.11, PyTorch 2.4.1 and OpenCV 4.10. The original 2D VxmPairwise forward pass succeeded, both regression tests passed, and both trained state dicts reloaded with identical predictions on the last batch.

Latest verified command: `python run.py reproduce`. Read `results/summary.csv`, or inspect per-case error and folding in `results/cases.csv`.

Next command: `python run.py demo --out scratch/demo`. For a full fresh run use `python run.py reproduce --out scratch/reproduction` and compare the regenerated table with the archived results. Real experiment limitations and test splits are documented in README.
