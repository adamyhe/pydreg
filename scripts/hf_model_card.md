---
license: gpl-3.0
tags:
- genomics
- bioinformatics
- pro-seq
- gro-seq
- transcription
- regulatory-elements
- svm
- random-forest
---

# dREG pretrained models

Pretrained weights for [dREG](https://github.com/Danko-Lab/dREG) (Danko Lab), a method for detecting active transcriptional regulatory elements (promoters and enhancers) from PRO-seq/GRO-seq nascent-transcription data.

These are the **original dREG model parameters**, extracted from the R package's distributed `.RData`/`.RDS` files and repacked as framework-agnostic [safetensors](https://github.com/huggingface/safetensors) -- no retraining was performed. See [`config.json`](./config.json) for the full parameter listing.

Used by [`pydreg`](https://github.com/adamyhe/pydreg), a from-scratch Python port of dREG's inference pipeline. `DREGModel.from_pretrained()` / `DREGPeakSplitForest.from_pretrained()` download and load these directly.

## Files

| file | model | description |
|---|---|---|
| `svm.model.safetensors.zst` | dREG SVR | RBF-kernel epsilon-SVR that scores a genomic position's regulatory potential (~[0, 1]) from a 360-dim multi-scale feature vector. 605,187 support vectors. |
| `rf.model.safetensors.zst` | peak-split forest | 500-tree random forest regression model used only during peak calling, to decide whether two adjacent local score maxima should be merged into one peak or split into two. |
| `config.json` | both | Human-readable model configuration (mirrors the metadata embedded in each safetensors file's header). |

Both `.safetensors.zst` files are zstandard-compressed safetensors -- `pydreg`'s loader decompresses them transparently, or use `zstd -d` manually.

## Provenance

- **SVR** -- extracted from `asvm.gdm.6.6M.20170828.rdata` ([Zenodo](https://zenodo.org/records/10113379)), the pretrained model bundled with dREG. The saved object (class `gtsvm`, trained via `Rgtsvm`) is field-compatible with `e1071`'s standard RBF epsilon-SVR dual form -- no re-training, only a format conversion.
- **Peak-split forest** -- extracted from `rf-model-201803.RDS`, bundled with the dREG R package (`dREG/inst/extdata/`), a `randomForest` regression forest used only in `peak_calling_rf.R`'s `find_rf_peaks()`/`split_peak()`.

Both files contain the exact weights the original R package ships and uses at inference time -- this repo hosts a format conversion, not a retrained or fine-tuned model.

## Usage

```python
from pydreg.models import DREGModel, DREGPeakSplitForest

svr = DREGModel.from_pretrained()            # downloads svm.model.safetensors.zst
rf = DREGPeakSplitForest.from_pretrained()   # downloads rf.model.safetensors.zst

scores = svr.predict(X)   # X: (n_queries, 360) features from pydreg.features
```

See `pydreg`'s documentation for the full pipeline: informative-position scanning -> multi-scale feature extraction -> SVR scoring -> RF-assisted peak calling -> FDR filtering.

## Citation

If you use these models via pydreg, please cite the pydreg preprint:

> He, A. Y., & Danko, C. G. (2026). pydreg: a fast Python package for identifying active cis-regulatory elements from nascent transcription. *bioRxiv*. https://doi.org/10.64898/2026.09.06.745329

Since these are dREG's own pretrained weights, please also cite the original dREG papers:

> Danko, C. G., Hyland, S. L., Core, L. J., Martins, A. L., Waters, C. T., Lee, H. W., Cheung, V. G., Kraus, W. L., Lis, J. T., & Siepel, A. (2015). Identification of active transcriptional regulatory elements from GRO-seq data. *Nature Methods*, 12(5), 433-438. https://doi.org/10.1038/nmeth.3329

> Wang, Z., Chu, T., Choate, L. A., & Danko, C. G. (2018). Identification of regulatory elements from nascent transcription using dREG. *Genome Research*, 29, 293–303. https://doi.org/10.1101/gr.238279.118

as well as the version number of pydreg that you used.

## License

GPL-3.0, matching the original dREG R package (`License: GPL-3` in its `DESCRIPTION`). These files are a format conversion of dREG's publicly distributed pretrained model weights, not a redistribution of its source code, but are licensed the same way as the rest of this port.
