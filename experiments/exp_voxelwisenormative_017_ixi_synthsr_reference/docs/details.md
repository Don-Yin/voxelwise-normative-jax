# Can the grey-matter reference be processed with a matched SynthSR route

This experiment needs the pinned external reference chain and tool installations described in experiment 000.

Set `FREESURFER_HOME`, `IXI_REFERENCE_CHAIN`, `IXI_T1_ARCHIVE` and `IXI_DERIVATIVES` explicitly.

Supply your prepared head manifest and its digest with `IXI_HEAD_MANIFEST` and `IXI_HEAD_MANIFEST_SHA256`.

Run `python run.py stage --workers 4` to check each crop against the supplied manifest.

The historical exact crop check used 57 retained public crops. This release includes none of those files.

Use a checked reference archive and pass explicit staged and published-input digests when using those routes.

SynthSR uses FreeSurfer 7.4.1's v1 model on CPU. The registered grey matter uses the same template and mask.

Compare independent public runs with `python run.py compare --run-dir <maps> --comparison-dir <controls>`.

The 12 comparison maps use the same SynthSR procedure on a second machine.

Supply `<controls>/<IXI-ID>/gm-modulated-s8.nii.gz` for each of the 12 public controls.

The code that generated the second-machine reference maps is not part of this repository.

Full Slurm jobs are described in `slurm/`. Verify quality before freezing a reference matrix.
