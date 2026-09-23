# Spectrum Aided Vision: Classification and Detection

Combined Smart India Hackathon repository for two desktop applications that apply Spectrum Aided Vision Enhancement (SAVE) to endoscopic imagery and then perform image classification or object detection.

> Repository name placeholder: replace `TeamName` and `SIHID` before submission.

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [System Workflow](#system-workflow)
5. [Applications](#applications)
6. [Models](#models)
7. [Dataset](#dataset)
8. [Repository Structure](#repository-structure)
9. [Installation](#installation)
10. [Running from Source](#running-from-source)
11. [Using the Applications](#using-the-applications)
12. [Windows Installers](#windows-installers)
13. [Reported Results](#reported-results)
14. [Documentation and Demo](#documentation-and-demo)
15. [Privacy, Safety, and Responsible Use](#privacy-safety-and-responsible-use)
16. [Known Limitations](#known-limitations)
17. [Team Information](#team-information)
18. [License](#license)

## Overview

The repository combines the supplied Spectrum Aided Vision Classification and Spectrum Aided Vision Detection projects in one organized SIH structure. It contains:

- A PyQt5 classification application.
- A PyQt5 object-detection application.
- The best supplied checkpoint for each workflow.
- Calibrated NumPy lookup tables used for SAVE conversion.
- SMART Lab branding for the application header and window icon.
- Inno Setup definitions and the supplied Windows installer packages.
- Consolidated technical documentation and presentation files.
- The available classification demonstration video.

Raw clinical data, patient records, annotations, and dataset archives are intentionally excluded.

## Problem Statement

Standard white-light endoscopic images may not expose all spectral features that are useful for computer-assisted analysis. Hardware-based narrow-band imaging can improve the visibility of selected tissue structures, but specialized equipment may not always be available.

The project investigates whether a calibrated software transformation can generate a SAVE representation from ordinary imagery and use that representation as input to classification and object-detection models.

## Proposed Solution

The system provides two Windows desktop applications:

- **Classification:** predicts the image category using the encrypted supplied checkpoint.
- **Detection:** locates and labels relevant regions using the supplied detection checkpoint.

Both applications support image-based processing and include controls for video or camera workflows. The transformed view and model result are presented through a graphical interface and can be saved for review.

## System Workflow

```text
Image, folder, video, or camera
                │
                ▼
      SAVE lookup-table conversion
                │
                ▼
 Classification or detection model
                │
                ▼
       Display, review, and export
```

The SAVE stage uses `nor_wli_nbi_table.npy` to perform a calibrated pixel transformation. Each application then loads the checkpoint from its corresponding folder under `Model/`.

## Applications

### Classification Application

Location: `SRC/Classification_App/`

Main capabilities:

- Load individual images.
- Load a folder of images.
- Load and process video.
- Use a connected camera.
- Display the original and transformed image.
- Run the encrypted classification model.
- Save generated output.

The source uses `Model/Classification/best.pt.encrypted` and requires the authorized password used by the original encryption workflow.

### Detection Application

Location: `SRC/Detection_App/`

Main capabilities:

- Load images or video.
- Use a connected camera.
- Apply the SAVE transformation.
- Run object detection.
- Display detected regions and class labels.
- Save generated output.

The source uses `Model/Detection/best.pt`.

### Branding

The supplied SMART Lab image is the current project branding:

- `SRC/Logo/SMART_Lab.png`
- `SRC/Logo/SMART_Lab.ico`

Copies are kept in each application's `assets/` folder because the applications load their logo and window icon from those runtime paths. Obsolete classification and detection logo folders were removed.

## Models

### Classification Model

```text
Model/Classification/
├── best.pt.encrypted
└── nor_wli_nbi_table.npy
```

- `best.pt.encrypted` is the best classification checkpoint supplied by the original project.
- `nor_wli_nbi_table.npy` is the related SAVE lookup table.
- No unencrypted `.pth` or `.pt` classification checkpoint was supplied.
- No classification training notebook or training script was present.

The checkpoint stores eight abbreviated labels exactly as follows:

`w1.n`, `w2.dys`, `w3.ca`, `w4.ev`, `w5.ec`, `w6.ero`, `w7.d`, `w8.s`

These abbreviations must not be expanded without the authoritative class dictionary.

### Detection Model

```text
Model/Detection/
├── best.pt
└── nor_wli_nbi_table.npy
```

- `best.pt` is the best detection checkpoint supplied by the original project.
- `nor_wli_nbi_table.npy` is the related SAVE lookup table.
- No detection training notebook or training script was present.

The checkpoint stores these labels exactly as supplied:

`Dysplaisa`, `SCC`, `Bleeding`, `Inflamed`, `Aisle`, `Cardia`, `Foreign body`, `Equipment`, `Word`, `Bubble`, `Reflective`

The spelling `Dysplaisa` is preserved from the checkpoint and requires confirmation before release.

## Dataset

The training and validation dataset is restricted and is not distributed with this repository. No raw clinical images, annotations, patient records, or dataset archive are included.

Dataset access, checkpoint labels, privacy restrictions, and contact guidance are documented in [Dataset/README.md](Dataset/README.md).

The source classification report describes a 1,074-image study involving Normal, Dysplasia, and SCC categories. The detection presentation contains conflicting dataset totals and class tables; those values must be reconciled against the authoritative dataset record before publication.

## Repository Structure

```text
SIH-TeamName-SIHID/
├── Dataset/
│   └── README.md
├── Model/
│   ├── Classification/
│   │   ├── best.pt.encrypted
│   │   └── nor_wli_nbi_table.npy
│   └── Detection/
│       ├── best.pt
│       └── nor_wli_nbi_table.npy
├── SRC/
│   ├── Classification_App/
│   ├── Detection_App/
│   ├── Logo/
│   ├── Installer/
│   └── Needs_Review/
├── Screenshot/
│   ├── Screenshots/
│   └── Demo_Video/
├── Documentation/
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

Only the root README and the required Dataset README are included. Additional folder-level README files were intentionally removed.

## Installation

### Requirements

- Windows 10 or Windows 11 is recommended for the desktop interface and installers.
- Python 3.10 or another version compatible with the listed dependencies.
- A working camera and appropriate permissions for live-camera use.
- Git LFS when cloning or publishing the repository.

### Python Environment

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Merged Python dependencies:

- `numba`
- `numpy`
- `opencv-python`
- `pycryptodome`
- `PyQt5`
- `torch`
- `torchvision`
- `ultralytics`

### Git LFS

Model, lookup-table, executable, installer, video, and Office/PDF artifacts are configured for Git LFS through `.gitattributes`.

```bash
git lfs install
git lfs pull
```

## Running from Source

Run these commands from the repository root.

### Classification

```bash
python SRC/Classification_App/app.py
```

Enter the authorized encrypted-model password when requested.

### Detection

```bash
python SRC/Detection_App/app.py
```

The copied source applications were adjusted to load models from the merged `Model/Classification/` and `Model/Detection/` locations. The original repositories were not modified.

## Using the Applications

1. Start the required application.
2. Select an image, folder, video, or camera input supported by that interface.
3. Allow the SAVE conversion to complete.
4. Run or review the classification/detection result.
5. Save the output to an appropriate local folder.
6. Remove patient-identifying information before sharing any output.

For reproducible evaluation, record the application version, checkpoint hash, input source, preprocessing settings, software environment, and output path.

## Windows Installers

Supplied files are organized under:

```text
SRC/Installer/
├── Classification/
│   ├── App.exe
│   ├── SAVE_Classification_App_Installer.exe
│   └── Spectrum-Aided-Vision-Classification.iss
└── Detection/
    ├── SAVE_Detection_App_Installer.exe
    └── Spectrum-Aided-Vision-Detection.iss
```

Important: the `.exe` installers were prebuilt in the original repositories. Replacing source images does not rewrite an already compiled executable. To place the SMART Lab logo inside an installed application, rebuild the application executable from the updated source/assets and then compile the corresponding `.iss` installer again. Test the result on a clean Windows machine before distribution.

## Reported Results

The following values were transcribed from the supplied presentations and were not independently recomputed while organizing the repository.

### Classification SAVE Accuracy

| Method | Reported accuracy |
| --- | ---: |
| SVM | 81.25% |
| CNN | 100% |
| VGG16 | 83% |
| Logistic Regression | 75% |
| MobileNetV2 | 87% |

### Detection SAVE mAP50

| Model | Normal | Dysplasia | SCC |
| --- | ---: | ---: | ---: |
| YOLOv3 | 51.2% | 45.8% | 71.3% |
| Scaled YOLOv4 | 51.8% | 48.6% | 68.8% |
| YOLOv6 | 58.1% | 55.8% | 76.8% |
| YOLOv10 | 75.3% | 50.5% | 88.9% |
| YOLO-NAS | 70.2% overall | Not reported | Not reported |

These comparisons appear in the source documentation; they do not establish clinical performance or regulatory suitability.

## Documentation and Demo

The `Documentation/` folder contains:

- `SIH_Documentation.pdf` — consolidated documentation.
- `SIH_Presentation.pptx` — consolidated editable presentation.
- Original classification and detection DOCX reports.
- Original classification and detection presentations.

The `Screenshot/Demo_Video/` folder contains the available classification demonstration video. No detection demonstration video or standalone application screenshots were supplied.

## Privacy, Safety, and Responsible Use

- Do not commit raw clinical data or patient identifiers.
- Use only authorized, de-identified inputs.
- Follow applicable institutional ethics approval, consent, privacy, and licensing conditions.
- Do not use model output as a substitute for qualified clinical judgment.
- Validate performance on the intended population and acquisition equipment.
- Document failure cases, uncertainty, and known dataset limitations.
- Confirm redistribution rights for models, branding, installers, reports, and data-derived artifacts.

## Known Limitations

- The dataset link and authoritative class dictionary were not supplied.
- The eight abbreviated classification labels require mapping confirmation.
- The detection label spelling `Dysplaisa` requires confirmation.
- Detection dataset statistics conflict across the supplied presentation.
- Training notebooks and training scripts were not present.
- Classification uses an encrypted checkpoint and requires its authorized password.
- No detection demo video or standalone screenshots were supplied.
- Reported metrics have not been independently reproduced.
- Existing compiled installers must be rebuilt to embed updated branding.

Detailed unresolved items and the source inventory are available under `SRC/Needs_Review/`.

## Team Information

- **Team Name:** `TeamName` — replace before submission
- **SIH ID:** `SIHID` — replace before submission
- **Institution:** To be added
- **Team Members:** To be added
- **Project Maintainer:** To be added
- **Contact:** To be added

## License

The repository includes an MIT License. Confirm ownership and redistribution permission for every supplied model, lookup table, document, logo, executable, installer, and third-party component before publishing the repository.

