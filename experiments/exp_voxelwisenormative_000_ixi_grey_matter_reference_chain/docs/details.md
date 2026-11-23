# Can we build a grey-matter reference using IXI scans alone

Install FSL 6.0.7.18, FreeSurfer 7.4.1 and the optional reference-chain code separately.

- Run [the downloader](../../../scripts/download-ixi.sh) to obtain the public IXI archive.
- Run [the preparation script](../../../scripts/preprocess-ixi.py) with the archive and demographic workbook.
- Supply the prepared directory as `IXI_PREPARED` and its manifest digest as `IXI_HEAD_MANIFEST_SHA256`.
- Set `IXI_REFERENCE_CHAIN` to the external reference code and put the FSL tools on `PATH`.
- Set `FSLDIR`, `FREESURFER_HOME` and `FSLOUTPUTTYPE=NIFTI_GZ` for that installation.
- For SynthStrip subprocess profiling, set `SLURM_PROFILE_BIN` to `scripts/slurm-profile.sh` expanded from the root.
- Run `python run.py prepare`, then `python run.py template --workers 4`, then `python run.py register`.
- Run `python run.py publish --derivatives <ixi-derivatives>` to assemble the input for experiment 001.

The pipeline skull-strips, segments, registers and smooths grey matter at 8 mm.

Template construction uses a symmetric cohort mean and a nonlinear registration pass.

The external chain is not bundled. New crops and templates may differ from the historical run across FSL builds.
