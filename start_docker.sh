#!/bin/bash
docker run -it \
  --gpus all \
  --device=/dev/video0:/dev/video0 \
  -p 8080:8080 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/projects/PIXIE:/workspaces/PIXIE \
  cobaltconcrete/pixie:cu121-torch230 \
  /bin/bash