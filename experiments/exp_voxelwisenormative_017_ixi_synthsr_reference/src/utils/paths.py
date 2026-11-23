"""Local and caller-supplied paths for the IXI SynthSR reference."""
import os
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS = EXPERIMENT_DIR.parent
RESULTS, INTERMEDIATES = (EXPERIMENT_DIR / name for name in ('.results', '.intermediates'))
HEAD_MANIFEST = Path(os.getenv('IXI_HEAD_MANIFEST', 'data/ixi/prepared/manifest.tsv'))
ARCHIVE = Path(os.getenv('IXI_T1_ARCHIVE', 'data/ixi/IXI-T1.tar'))
STAGED_INDEX = RESULTS / 'manifests' / 'staged-heads.json'
PUBLISHED = Path(os.getenv('IXI_DERIVATIVES', 'data/ixi/derivatives'))
REFERENCE_REPO = Path(os.getenv('IXI_REFERENCE_CHAIN', 'reference-chain-not-configured'))
FS_HOME = Path(os.getenv('FREESURFER_HOME', 'freesurfer-not-configured'))
