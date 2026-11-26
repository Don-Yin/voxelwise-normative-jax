# licences

## code

| component | terms | source |
|---|---|---|
| this package and its experiment code | MIT | [licence text](../LICENSE) |
| PCNtoolkit 1.4.0 | GPL-3.0-only; optional, separately installed, not bundled | [release metadata](https://pypi.org/project/pcntoolkit/1.4.0/) |
| FSL and FreeSurfer | separate terms for externally installed imaging tools | [FSL licence](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html) · [FreeSurfer licence](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense) |

## data

| dataset | terms | source |
|---|---|---|
| IXI — Information eXtraction from Images (EPSRC GR/S21533/02), Brain Development | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | [IXI Dataset](https://brain-development.org/ixi-dataset/) |

the [speed race GIF](assets/speed-race.gif) is adapted from Brain Development's [IXI Dataset](https://brain-development.org/ixi-dataset/) and is licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); the share-alike terms apply to that rendered image file, not the MIT-licensed code. no scans, fitted arrays or z-score maps are distributed as data files.

## figure marks

| mark | use or permission | source |
|---|---|---|
| voxel-brain icon | project artwork, used for the banner and social card | [icon](figures/logo/icon.png) |
| [PCNtoolkit icon](assets/pcntoolkit-icon.png) | [GPL-3.0](assets/pcntoolkit-gpl-3.0.txt); side padding cropped | [upstream icon](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/blob/e9469dc70ad761301b3e970b5585f167373203db/doc/_static/pcn-icon.png) · [upstream licence](https://github.com/predictive-clinical-neuroscience/PCNtoolkit/blob/e9469dc70ad761301b3e970b5585f167373203db/LICENSE) |
| blue filled circle | measured JAX result from this package | [plot source](figures/result/standard.py) |
| grey open diamond | measured JAX comparison baseline, including the single A100 in exp_018 | [plot source](figures/result/runtime.py) |
| grey plus sign | ideal four-device time in exp_018, calculated from the single-device time | [plot source](figures/result/runtime.py) |

the upstream PCNtoolkit repository uses GPL-3.0. we found no separate icon terms.

cropping the transparent side padding on 2026-11-23 changed 460 × 460 px to 385 × 460 px.

the MIT licence applies only to this repository’s own work.

names and marks identify the software and hardware used; they do not imply endorsement.
