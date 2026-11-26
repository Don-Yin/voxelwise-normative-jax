"""Shared layout and aggregate table readers for the public plots."""

from contextlib import contextmanager
import csv
from hashlib import sha256
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
BLUE = "#365b88"
RED = "#a33a42"
TEAL = "#16766f"
PACKAGE = {"marker": "o", "color": BLUE, "ms": 5}
PCNTOOLKIT = {"marker": "s", "color": RED, "ms": 5, "mfc": "white", "mew": 1.3}
JAX_BASELINE = {"marker": "D", "color": "#666666", "ms": 5, "mfc": "none", "mew": 1.1}


def rows(name):
    """Read a published aggregate CSV, independently of the working directory."""
    with (DATA / name).open(newline="", encoding="utf-8") as table:
        return list(csv.DictReader(table))


def provenance(job, *tables, **details):
    """Name and bind each committed table to the builder's scientific job."""
    sources = {f"data/{name}": sha256((DATA / name).read_bytes()).hexdigest() for name in tables}
    return {"scientific_job": job, "table_sha256": sources, **details}


def read_json(name):
    """Read a style or aggregate receipt object, naming the input on failure."""
    source = DATA / name
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read figure input {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in figure input {source}")
    return value


@contextmanager
def canvas(*args, width, height, **kwargs):
    """Build the figure at its display size using the shipped style settings."""
    import matplotlib.pyplot as plt

    with plt.rc_context(read_json("plot-style.json")):
        yield plt.subplots(*args, figsize=(width / 2.54, height / 2.54), **kwargs)


def panel(ax, title, *, grid="y"):
    """Use left-aligned panel names and fine dotted linear-axis guides."""
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5, pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color="#d9dfe4", ls=":", lw=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=3)
    ax.minorticks_off()


def categories(ax, ticks, labels, *, pad):
    """Left-align categorical y labels at one edge."""
    ax.set_yticks(ticks, labels, ha="left")
    ax.tick_params(axis="y", length=0, pad=pad)


def key(fig, handles, labels, *, x=0.5, y=0.99, columns=None):
    """Place a compact legend in its own band."""
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(x, y),
               ncol=columns or len(handles), fontsize=9, handlelength=1.3,
               handletextpad=0.55, columnspacing=1.6, frameon=False)
