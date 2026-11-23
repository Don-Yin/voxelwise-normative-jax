# Which original flags recur when the reference group is resampled

The original flag rule is strict `abs(z) > 1.96`. Only original flags are in the recurrence denominator.

Each draw resamples the training rows with replacement; the test scan remains held out.

The program recomputes response scaling for each draw and fits the same float64 model.

Run `slurm/bootstrap-a100.sh` with the verified IXI matrix and converged reference artifacts staged locally.

`python run.py --live --trials 8` serves a local view on a GPU machine; synthetic tests require no data.

The result receipt links a packed flag archive. Use `python run.py --verify --run-name <run>` to check the run.

This measure describes sensitivity to reference sampling, not a diagnosis.
