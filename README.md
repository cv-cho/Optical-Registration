# Orbit T2 COR CT/MRI Registration Workbench

This repository contains only the registration program. Patient image data and generated annotations must not be committed.

Korean registration workflow manual: [REGISTRATION_MANUAL_KO.md](REGISTRATION_MANUAL_KO.md)

## Expected Layout

After cloning, unpack the collaborator data archive into the repository root so these paths exist:

```text
data/
reports/series_inventory_series.csv
reports/ct_axial_1mm_candidates.csv
outputs/landmarks/work_queue.csv
```

The data archive uses relative paths, so the program can run from any clone path.

## Setup on Windows

```powershell
cd <clone-root>
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Setup on macOS

Python 3.11 or 3.12 is recommended.

```bash
cd <clone-root>
bash scripts/setup_macos.sh
```

If `python3` points to the wrong interpreter:

```bash
PYTHON_BIN=/opt/homebrew/bin/python3.12 bash scripts/setup_macos.sh
```

## Verify Data Package

Windows:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_data_package.py
```

macOS:

```bash
bash scripts/verify_data_package.sh
```

## Start Registration

Windows:

```powershell
.\.venv\Scripts\python.exe .\scripts\launch_dual_landmark_workbench.py
```

macOS:

```bash
bash scripts/run_workbench.sh
```

To open a specific patient:

Windows:

```powershell
.\.venv\Scripts\python.exe .\scripts\launch_dual_landmark_workbench.py --patient-id 102059
```

macOS:

```bash
bash scripts/run_workbench.sh --patient-id 102059
```

Default MRI series is `T2 COR dixon_(IN W)_in`. The left panel starts as CT axial and the right panel starts as MRI coronal.

The top `Done` counter shows how many patients in the current work queue have a saved globe registration transform.

Axial and coronal views show patient-side `L`/`R` labels in the top corners. Globe surface clicks in axial/coronal views automatically save the side from the DICOM LPS coordinate, so the manual eye selector is only a fallback for sagittal or unusual cases.

Use `Ctrl+=` / `Ctrl++` to zoom in, `Ctrl+-` to zoom out, and `Ctrl+0` to reset zoom on the last hovered/clicked view. On macOS, the same shortcuts work with `Command` instead of `Ctrl`. While zoomed, a small overview appears in the top-left of the view; drag the yellow ROI box to pan the zoomed region. `Ctrl` + mouse wheel also zooms, and the normal mouse wheel still changes slices.

If a fitted globe center is wrong, select `CT LC`, `CT RC`, `MRI LC`, or `MRI RC` in the `Center` control and press `Set center by click`, then click the correct center in the matching native CT/MRI view. After that click, the tool returns to normal globe surface point mode; press `Esc` to cancel before clicking. Existing center markers can also be dragged to create/update a forced center. Forced centers are marked with `*`; right-click a forced center or use `Clear forced center` to remove the override.

## Outputs

Annotations and transform outputs are written under:

```text
outputs/landmarks/
```

Send back at least:

```text
outputs/landmarks/annotations.sqlite
outputs/landmarks/<patient_id>/
```

Do not commit `data/`, `reports/*.csv`, or `outputs/landmarks/*`.
