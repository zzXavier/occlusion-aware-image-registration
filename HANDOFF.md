# P09 handoff

Local run `cpu-1769` completed on Windows, Python 3.11, PyTorch 2.4.1 and OpenCV 4.10. The original 2D VxmPairwise forward pass succeeded, both regression tests passed, and both trained state dicts reloaded with identical predictions on the last batch.

Latest verified command: `python run.py reproduce`. Read `results/summary.csv`, or inspect per-case error and folding in `results/cases.csv`.

Next command: `python run.py demo --out scratch/demo`. For a full fresh run use `python run.py reproduce --out scratch/reproduction` and compare the regenerated table with the archived results. Real experiment limitations and test splits are documented in README.
