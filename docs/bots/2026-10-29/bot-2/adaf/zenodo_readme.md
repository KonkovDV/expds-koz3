# ADAF – Model Weights

This repository contains model weights for the ADAF (Automatic Detection of Archaeological Features) tool. The models are in PyTorch format and are intended for use with the [AiTLAS Python package](https://github.com/biasvariancelabs/aitlas), which is built on top of PyTorch.

Instructions for using these models within the ADAF framework are available on the official GitHub repository: [adaf](https://github.com/EarthObservation/adaf/releases/tag/v1.0.0)

---

## Object detection (Faster R-CNN)

These models detect archaeological features using bounding boxes.

Files list:
* OD_AO.tar
* OD_barrow.tar
* OD_enclosure.tar
* OD_ringfort.tar


## Semantic segmentation (HRNet)

These models generate pixel-wise segmentations of archaeological features.

Files list:
* AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar
* barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar
* enclosure_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar
* ringfort_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar

---

## Class Definitions
Each model corresponds to one of the following archaeological classes:

* AO (All archaeology): A general-purpose model trained on all feature types combined as a single class.
* Barrows: Models specifically trained to detect barrow monuments.
* Enclosures: Models for detecting archaeological enclosures.
* Ringforts: Models specialised in identifying ringfort structures.

These classes, their definitions, and usage guidance are described in detail in the ADAF user manual, available on GitHub.

---

All models were trained using the [AiTLAS library](https://github.com/biasvariancelabs/aitlas).