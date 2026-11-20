<p align="center">
  <a href="https://github.com/KonkovDV/expds-koz3/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/KonkovDV/expds-koz3/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&amp;logoColor=white">
  <img alt="PyTorch 2.11" src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&amp;logoColor=white">
  <img alt="Rust core" src="https://img.shields.io/badge/Rust-core-000000?logo=rust&amp;logoColor=white">
  <a href="Dockerfile"><img alt="Docker" src="https://img.shields.io/badge/Docker-pantum%2Fn7q-2496ED?logo=docker&amp;logoColor=white"></a>
  <img alt="Task No. 3, Complexing" src="https://img.shields.io/badge/Task_No._3-Complexing-7A4E2D">
  <a href="LICENSE"><img alt="All rights reserved" src="https://img.shields.io/badge/license-all_rights_reserved-555555"></a>
</p>

[Русский](README.md) | English

# StrataTrace

Archaeological object detection from lidar, imagery and geophysics. A solution to task contest No. 3
"Complexing" of the "Expedition. Data Science" technology contest.

Input is a sample folder: LAS point clouds, satellite scenes, orthophotos, aerial photographs from the 1940s,
magnetic, electrical resistivity and thermal survey maps. Output is one GeoJSON in EPSG:3857 with objects of
seven classes: damaged and intact burial mounds, hillforts, architecture, fortifications, geophysical area
anomalies and finds. A full run fits into 30 minutes on one RTX 4090.

**For the jury:** [description in the contest template](docs/README.md) (Russian) ·
[technical report](docs/TECHNICAL_REPORT.md) (Russian) · [AI declaration](docs/ai_declaration.md) ·
[archive manifest](docs/final_archive.sha256) · [weights](models/README.md) ·
[ConvNeXt training](training/lidar_convnext/README.md)

## Contents

- [Where to look](#where-to-look)
- [In short](#in-short)
- [How it works](#how-it-works)
- [Why it is built this way](#why-it-is-built-this-way)
- [How every change was accepted](#how-every-change-was-accepted)
- [Running](#running)
- [Archive and reproducibility](#archive-and-reproducibility)
- [Hard questions](#hard-questions)
- [Layout](#layout)
- [License](#license)

## Where to look

1. What the solution does and how to run it: [`docs/README.md`](docs/README.md), in the contest template.
2. The metric, the method and the case for every edit: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).
3. That the repository matches the submitted archive: `python scripts/verify_archive.py --local` and
   [`docs/final_archive.sha256`](docs/final_archive.sha256).
4. Code: entry point [`inference/main.py`](inference/main.py), run [`expds/pipeline.py`](expds/pipeline.py),
   post-processing [`expds/watchdog.py`](expds/watchdog.py).

## In short

| | |
|---|---|
| Final version | third final attempt, 19.11.2026, 22:29 Moscow time; config [`configs/default.toml`](configs/default.toml) |
| Solution archive | 178 files, 1 365 346 122 bytes, sha256 `171bed3024aa6b55252c0f6b70116952e815a806bb29271d0f26e9737f8b7252` |
| Archive build | byte for byte from this repository, the weights and PyPI packages |
| Lidar mounds | ensemble of 10 U-Nets with a ConvNeXt-T encoder and a CenterNet head, 5 and 12 relief channels, D8 averaging |
| Other classes | U-Net ResNet-18 per survey type; magnetic finds by pixel-level LightGBM |
| Thresholds | official metric, cross-validation with each site family in one fold |
| Time | 1500 s budget; a watchdog writes the result 3 minutes before the 30-minute limit |
| Tests | 551 tests on synthetic data, CI on GitHub |

The organizers send closed-set scores to teams as confidential information (technical regulations,
clause 8.5). They will appear here once the final ranking is published.

## How it works

```
sample folder ─┬─ geophysics: MagGr pixel boosting · EtGr isolines and boosting · TerFl anomalies
 (catalog, CRS) ├─ lidar: 0.1 m DTM → hillfort and fortification masks, hillfort shape rule
                ├─ lidar ConvNeXt: 10 networks × D8 → mounds
                ├─ imagery: satellite, orthophoto, aerial — U-Net ResNet-18
                └─ crop check: a satellite mound is rescored on a 96 px window
                                       ↓
        post-processing: group thresholds → merge within survey type → 0.5 m mound circles
                         → 0.05 m micro-disks → geometry check → regulation limits → GeoJSON
```

| Survey | Classes | Method |
|---|---|---|
| Lidar | mounds, hillforts, fortifications | DTM from ground points with Delaunay fill; ConvNeXt ensemble for mounds, U-Net masks on a 1 m DTM, relief σ channel |
| Satellite | mounds, hillforts, architecture | centre map with focal loss, hillfort mask, architecture circles |
| Orthophoto | architecture | architecture network shared with satellite |
| Aerial | mounds, fortifications | fortification mask; a lidar mound is copied into the aerial layer where the photo covers it |
| Magnetic | finds, area objects | 60 multi-scale map features, six LightGBM models; noise and boundary masks from `extra/` |
| Resistivity | area objects | isolines and LightGBM; each slice and profile is its own layer |
| Thermal | area objects | anomalies against the background |

The DTM, hillshade and peak core is written in Rust; NumPy versions of the same functions give the same
result, checked by `tests/test_accel.py`.

## Why it is built this way

**Late fusion.** The official metric counts a mound marked on lidar and on a photo as two references. So each
survey type runs its own branch with its own confidence scale and thresholds, and objects from different survey
types are never merged. A duplicate within one survey type is a false positive, so those are merged.

**A mound is a 0.5 m circle, a polygon a 0.05 m disk.** The metric scores a hit by coverage and by the
prediction's interior point, not by its outline. A disk at the interior point of a hit polygon is a hit too,
and a prediction whose interior point lies in the reference with less than half coverage becomes a hit.
This is claim 2 of the report, with a proof; in cross-validation the micro-disks gave +0.00197 with a 0.9995
probability of improvement.

**Gray MagGr threshold 0.22.** The training optimum is 0.3246, but the model is less confident on maps
rendered differently. The threshold is the minimax-regret choice over two regimes: −0.0027 in the worst case,
up to +0.012 under a shifted scale (report, section 5.2).

## How every change was accepted

- Thresholds: official metric over five folds; a site family (first word of the site name) stays in one
  validation fold.
- Comparison: paired bootstrap over families, 2000 resamples. A change is accepted at a gain of at least +0.002
  with probability at least 0.8 and a non-negative gain on 34 new sites.
- An addition to a class is checked by the exact metric condition: a block of *a* hits and *b* false positives
  raises the class F1 if and only if *a*/(*a* + *b*) > *F*/(4 − *F*) (claim 1 of the report).
- These rules rejected aerial mound detectors (YOLO, D-FINE and others) and a lidar false-mound filter: their
  gains on old sites did not carry over to new ones (report, section 8).

## Running

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

```bash
python inference/main.py SAMPLE_FOLDER result.geojson
```

```bash
python -m expds.writer check result.geojson
```

Docker: image `pantum/n7q:20261024`, built from the [`Dockerfile`](Dockerfile); code and weights are mounted:

```bash
docker run --rm --gpus all -v "$(pwd)":/app -v /path/to/sample:/data/input -v "$(pwd)/out":/data/output \
  pantum/n7q:20261024 python inference/main.py /data/input /data/output/result.geojson
```

Without a GPU the solution runs on CPU, slower; when time runs short, a soft ceiling skips optional steps and
the watchdog writes what has been found.

## Archive and reproducibility

```bash
python scripts/fetch_thirdparty.py
```

```bash
python scripts/verify_archive.py --local
```

```bash
python scripts/build_submission.py --out dist/submission.zip
```

```bash
python scripts/verify_archive.py dist/submission.zip
```

The weights are trained on contest data and are not in git; they come from the final archive
([`models/README.md`](models/README.md)). Third-party packages are downloaded from PyPI with sha256 checks. The
build is deterministic: the same files and weights give the same archive to the byte. Training of every network
is described in [`docs/README.md`](docs/README.md) and
[`training/lidar_convnext/README.md`](training/lidar_convnext/README.md).

```bash
pytest -q
```

## Hard questions

**Is this the team's own code?** Yes. Third-party components — Relief Visualization Toolbox, delaunator,
lleaves and llvmlite — are kept unchanged or with one documented edit, with their licenses. The solution code is the
team's own and does not depend on any other solution.

**Were thresholds tuned on the public set?** No. One exception is disclosed in the report: for the gray MagGr
threshold, the confidence regime was estimated from the geophysics precision and recall in the platform log.

**Does it look up known monuments?** No. There are no monument lists, site names or coordinates used as
features; objects are found from the input data only.

**Is the result deterministic?** Networks run in FP32 with deterministic convolutions, LightGBM with
`deterministic = true`. On a very slow machine the soft ceiling may skip optional steps.

**How is the coordinate system chosen?** A raster uses its own CRS; `UTM.json` only when there is none or it
does not convert to EPSG:3857. For LAS clouds the header CRS and `UTM.json` are compared, and the one that puts
the cloud near the meridian of its zone wins.

**Were AI tools used?** Yes, they are listed in the [declaration](docs/ai_declaration.md) under clause 8.5.3.
They are not called during inference and are not in the archive.

## Layout

```
.
├── inference/main.py          single entry point
├── expds/                     pipeline: catalog, CRS, lidar, imagery, geophysics, post-processing, writer
├── centerhead/                mound head decoding
├── configs/default.toml       thresholds, weights and flags of the final version
├── rust/expds_core/           Rust core sources
├── scripts/                   archive build and checks, training of auxiliary models
├── training/lidar_convnext/   ConvNeXt ensemble training
├── models/                    weight descriptions and checksums
├── materials/                 official contest metric
├── tests/                     tests on synthetic data
└── docs/                      description, technical report, AI declaration, archive manifest
```

## License

The repository is open for reference; it grants no license to the code, all rights reserved. The exclusive
right in the solution must remain transferable to the contest operator. Terms are in [`LICENSE`](LICENSE).
Contest data and weights are not in the repository.
