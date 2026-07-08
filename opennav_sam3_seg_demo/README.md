# Segmentation demo using SAM3

This demonstration works entirely in a Docker container with no extra installation or setup.

## Build the Docker image

Navigate to the Docker compose file location inside the `opennav_sam3_seg_demo/docker` directory and then build there.

```bash
export RENDER_GID=$(getent group render | cut -d: -f3)
docker compose build
```

NOTE: `render` user group is automatically created during AMD ROCm installation. If not, please install ROCm before building the image to set the right `GID` for `render` group. Without correctly setting the `render` group, the container cannot access the GPU!

This will take a long time...

## First time setup before running

Set the location for downloading and storing the SAM3 model weights, and build artifacts on the host machine.

For example:

```bash
mkdir -p ~/opennav-sam3-inference-model
```

Then set that path to `SAM3_MODEL_DIR` environment variable.

```bash
export SAM3_MODEL_DIR=~/opennav-sam3-inference-model
```

Even better, add that line to `~/.bashrc` to make it persistent between runs.

And allow the container to access host's X-server for GUI

```bash
xhost +local:
```

## Run the demo

```bash
docker compose run --rm sam3-demo
```

Two windows should pop up, one for setting the prompts and another one displays the segmented frames.
The image display window can be toggled to fullscreen by pressing the `F` key.