# Install Dependencies and Setup SAM3

Install Docker (with BuildKit, usable without `sudo`). SAM3 runs in a ROCm 7.14 / MIGraphX 2.17 Docker image and needs a `gfx1151` GPU (Strix Halo) with access to `/dev/kfd` and `/dev/dri`.

Then clone this repository in your workspace source directory.

Create a folder on your computer for storing the model weights and build artifacts, and set this directory to `SAM3_MODEL_DIR` environment variable. For default setup:

```bash
mkdir -p ~/opennav-sam3-inference-model
export SAM3_MODEL_DIR=~/opennav-sam3-inference-model
```

To make this setting more persistent, add the export command to `~/.bashrc`.

Change to `opennav_sam3_setup` directory inside the `src` directory of your workspace.

```bash
cd <workspace-path>/src/opennav_amd_semantic_navigation/opennav_sam3_setup
```

Then run the `setup.sh` script in that directory to build the SAM3 Docker image and get the model weights:

```bash
./setup.sh
```

Without any modifiers, the script builds the Docker image and downloads the model weights to `SAM3_MODEL_DIR` folder.

>Note: The setup script provides two options to either obtain SAM3 weights from the official repository or from a community mirror, either of them works fine.

Then build model artifacts for a specific resolution/pipeline which takes a few minutes.

```bash
# Text-prompt (~18 min @504px)
./docker/run.sh python export/build.py --pipeline text --imgsz 504

# Both resolutions (~45 min total)
./docker/run.sh python export/build.py --pipeline text --imgsz 504 1008
```

The script outputs files in the same `SAM3_MODEL_DIR` under `onnx_files_*` folder.

## Setting up without the environment variable

If the environment variable for storing model weights and artifacts isn't set, the scripts will output within the `opennav_sam3_setup` directory that can later be moved to another location.
