# opennav_amd_semantic_segmentation








## Build and Deploy

Use the following commands to build the docker image for the AMD Ryzen AI Max. This will create the Docker container with the necessary ROCm and PyTorch dependencies with ROS 2 Jazzy, then move the packages in this repository and build them.

This assumes your `HF_TOKEN` is set as an environmental variable (from your `~/.bashrc` for instance)

On `docker run`, it'll automatically bring up the SAM3 segmentation node as the entrypoint.

```bash
cd ~/ros2_ws/src/opennav_sam3_inference
docker build -t opennav_sam3_inference .

docker run --rm -it \
    --network host \
    --ipc host \
    --shm-size 16G \
    --device=/dev/kfd --device=/dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -e HF_TOKEN=$HF_TOKEN \
    --name sam3 \
    opennav_sam3_inference
```

## Development

Use this Docker image, but run with `bash` to enter the container rather than immediately inferencing. Mount your workspace with `-v` to persist changes. See Nav2's tutorial on [Docker Zero to Hero](https://docs.nav2.org/tutorials/docs/docker_dev.html) to learn more.


## Ryzers

```bash
git clone https://github.com/AMDResearch/Ryzers
source rai17env/bin/activate # use venv with AMD-specific pytorch, ONNX, and ROCm libs
pip install Ryzers/  # install deps 
```

While waiting, Create a hugging face account and
* Accept license at https://huggingface.co/facebook/sam3 and
* Get token from https://huggingface.co/settings/tokens 
* Update packages/vision/sam3/config.yaml with your `HF_TOKEN`

once done:
```bash
ryzers build sam3  # Installs docker images containing SAM weights
ryzers run # May take some time to get access from repo authors, then some time to download the weights
# ryzers run bash to enter the container to run arbitrary code
```

TODO
- Download weights once so it doesn't each time.  Fix it by mounting a host directory as the HF cache. You can also pre-download outside the container from your rai17env once access is granted
- MOre detailed instructions & into scripts to run
- TODO performance
  - .half() / torch.autocast(dtype=torch.float16) or bfloat16 — roughly
  2× throughput on ROCm/CUDA, usually with no visible quality loss for
  segmentation. Single biggest free win for live use.
  - torch.compile(model) — another 10–30% on recent PyTorch, though first
   call is slow (graph capture).
  - torch.inference_mode() is slightly faster than no_grad() — strict
  superset for pure inference.
  - Batching frames/prompts through the processor — if you ever get
  multi-camera input, batching is a real speedup.

  - Image reslution


### Fix Segmentation fault on startup with correct rocm version

```bash
docker build \
  --build-arg BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  -t ryzerdocker \
  packages/vision/sam3/

docker run -it --rm --shm-size 16G \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN=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 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  ryzerdocker

huggingface-cli login --token $HF_TOKEN
python3 test_sam3.py
```
