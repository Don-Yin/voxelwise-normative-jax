# Does batching resamples speed up each refit on one device

Run the generated CPU fixture to check the sweep and receipt schema:

```sh
JAX_PLATFORMS=cpu python run.py sweep --synthetic --precision float64
```

The real sweep uses one A100 and verified IXI inputs. It needs a Linux Slurm or Hugging Face job.

Each pass processes the same 64 reference draws; batch size grows from 1 to 64 or device memory exhaustion.

Compare z-score differences and threshold flips against the batch-size-one reference.

Timing distinguishes device execution, host collection and compilation. The trace samples device activity.
