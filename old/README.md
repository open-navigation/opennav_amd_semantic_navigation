
# OLD Ryzers

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
