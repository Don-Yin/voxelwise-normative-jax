# Hugging Face Jobs

The helpers stage files, render commands and check results for experiments 004, 010, 011 and 013. The renderer does not submit anything.

```bash
python -m shared.hugging_face_jobs.build_input --profile synthetic "$SYN_INPUT"
python -m shared.hugging_face_jobs.jobs --synthetic-input "$SYN_INPUT" \
  --ixi-input "$IXI_INPUT" --bucket "$NAMESPACE_AND_BUCKET" --only mechanism-timing
```

Pass your own `namespace/name` as `--bucket`. The staged directory must be new or empty. The synthetic bundle contains only source and configuration: the worker generates its input arrays.

For jobs using IXI, explicitly select `--profile ixi` to stage the verified public artifacts from a local build. Check the dataset licence and review the files before transferring them to any compute service. The container verifies its input inventory, requires a real job identifier for an IXI run, and writes results only to the configured output mount.

To verify results held in a local folder, use `python -m shared.hugging_face_jobs.fetch <run> --source <folder>`. To fetch from your own bucket, use `--bucket <namespace/name>` in place of `--source`.

Run the local command and receipt tests with `python -m pytest shared/hugging_face_jobs/tests`.
