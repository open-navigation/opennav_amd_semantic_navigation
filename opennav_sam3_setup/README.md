# Install Dependencies and Set Up SAM3

OpenNav continues to run on the host with ROS 2 Jazzy, Conda and colcon. The
supported SAM3 stack uses ROCm 7.14 packages at Debian release `7.14.1-0`,
MIGraphX 2.17, ONNX Runtime 1.24.2, SciPy 1.17.1 and Python 3.12; no ROS or
runtime container is required. The `rocm_sdk` compatibility module reports
API version 7.14.0, as in the validated runtime.

Download a recent version of `conda` / `miniforge`. Check out [miniforge repo](https://github.com/conda-forge/miniforge) for installation steps. 

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

Then run the `setup.sh` script in that directory to set up Python, ROCm, and
the other runtime dependencies. Model weights remain an optional, separately
licensed download:

```bash
./setup.sh
```

Run the script as your normal login user, not with sudo. It invokes sudo only
for ROCm and a MIGraphX prefix below /opt, installs that prefix as root-owned
and read-only to non-root users, and keeps a custom user-writable
SAM3_MIGRAPHX_ROOT owned by the invoking user. Before making changes it verifies
read/write access to /dev/kfd and at least one /dev/dri/renderD* node; add the
user to the render and video groups and log in again if that check fails.

Without modifiers, the script installs the pinned ROCm 7.14 system runtime,
downloads checksum-verified MIGraphX and ONNX Runtime binaries, creates the
`opennav-sam3-inference` Conda environment, installs the bundled model config
and tokenizer files, and offers to download a verified checkpoint into
`SAM3_MODEL_DIR`. It does not compile MIGraphX or ONNX Runtime. In a
non-interactive shell, weight download is skipped unless `--yes` is supplied;
use `--skip-weights` to make that choice explicit.
Corrupt files in the script-managed download cache are removed and downloaded
again. Explicit MIGRAPHX_ARCHIVE and ORT_WHEEL_PATH files are never deleted
when their checksum is invalid.

The MIGraphX archive is installed in the isolated prefix
`/opt/opennav-sam3/runtime/0.2.0-rc4/migraphx`; it is not copied over the
system ROCm installation and is not registered with global `ldconfig`.

### Upgrading a legacy ROCm installation

The ROCm 7.14 multi-arch packages require `/opt/rocm` to be a directory that
contains `core-7.14`. If `/opt/rocm` is still an alternatives symlink to an
older tree such as `/opt/rocm-7.2.0`, `setup.sh` stops before running APT or
removing anything. Remove or migrate that legacy ROCm installation using the
system package manager, confirm that `/opt/rocm` is no longer a legacy
symlink, and rerun setup. The script intentionally never deletes an existing
ROCm installation automatically.

An environment created by the older ROCm 7.2 setup is deliberately not
modified in place. Replace that dedicated environment explicitly:

```bash
./setup.sh --recreate-env
conda deactivate
conda activate opennav-sam3-inference
```

If that environment is currently active, deactivate it before using
`--recreate-env`; setup refuses to delete an active environment.

Custom `SAM3_ROCM_PATH` and `SAM3_MIGRAPHX_ROOT` values are stored in that
environment's local runtime configuration and are restored on each activation.
When launching outside that Conda environment, pass the same overrides in the
environment of `ros2 launch`; the default fixed paths require no extra step.

The supported checkpoint is the gated official `facebook/sam3`
`model.safetensors` at revision
`3c879f39826c281e95690f02c7821c4de09afae7`, with SHA256
`6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`.
The non-gated download used by `./setup.sh --yes` is pinned to historical
revision `fe5e2ae858f82b44a0b18e80aa09fe1d5ed75a0b` of `1038lab/sam3`.
Its `sam3.safetensors` is byte-identical to the official checkpoint and is
verified against the same SHA256 before being installed. Do not use that
mirror's mutable `main` branch, whose current checkpoint is different.

To use the gated official source instead of the verified mirror:

```bash
./setup.sh --skip-weights
conda activate opennav-sam3-inference
hf download facebook/sam3 model.safetensors \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-dir "${SAM3_MODEL_DIR}/sam3"
echo "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a  ${SAM3_MODEL_DIR}/sam3/model.safetensors" \
  | sha256sum --check
```

Then build model artifacts for a specific resolution/pipeline. MIGraphX MXR
files and ORT caches are ABI-specific: do not reuse artifacts created with
MIGraphX 2.15 or 2.16. Build into a new, empty artifact directory.

```bash
conda activate opennav-sam3-inference

# Text-prompt (~18 min @504px)
python export/build.py --pipeline text --imgsz 504

# Both resolutions (~45 min total)
python export/build.py --pipeline text --imgsz 504 1008
```

The script outputs files in the same `SAM3_MODEL_DIR` under `onnx_files_*`.
The checkpoint remains a separate upstream-licensed download; setup does not
download a model bundle or Docker image.

## Setting up without the environment variable

If the environment variable for storing model weights and artifacts isn't set, the scripts will output within the `opennav_sam3_setup` directory that can later be moved to another location.
