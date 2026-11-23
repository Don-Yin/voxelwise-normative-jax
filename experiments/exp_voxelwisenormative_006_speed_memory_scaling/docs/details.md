# How does the number of voxels affect time and device memory

Each GPU case has three warm timed passes and reports live allocator peak bytes.

Reserved GPU pool and process resident memory are different measurements.

`slurm/scaling-a100.sh` runs both precisions and nine widths using the IXI artifacts.

`python run.py pcntoolkit-memory` uses generated responses with the matching train and test row counts.

Run `python run.py collect --help` to combine your GPU and reference measurements.

The resulting aggregate table is a local input to experiment 010.
