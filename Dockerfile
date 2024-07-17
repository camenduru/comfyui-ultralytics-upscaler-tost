FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 transformers==4.41.2 accelerate==0.30.1 insightface==0.7.3 onnxruntime==1.18.0 onnxruntime-gpu==1.18.0 \
	ultralytics==8.2.27 segment-anything==1.0 piexif==1.1.3 qrcode==7.4.2 requirements-parser==0.9.0 rembg==2.0.57 rich==13.7.1 rich-argparse==1.5.1 matplotlib==3.8.4 pillow==10.3.0 spandrel==0.3.4 && \
	git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
	git clone https://github.com/Extraltodeus/ComfyUI-AutomaticCFG /content/ComfyUI/custom_nodes/ComfyUI-AutomaticCFG && \
	git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts /content/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts && \
	git clone https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes /content/ComfyUI/custom_nodes/Derfuu_ComfyUI_ModdedNodes && \
	git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack /content/ComfyUI/custom_nodes/ComfyUI-Impact-Pack && \
	git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack /content/ComfyUI/custom_nodes/ComfyUI-Inspire-Pack && \
	git clone https://github.com/kijai/ComfyUI-KJNodes /content/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
	git clone https://github.com/Fannovel16/comfyui_controlnet_aux /content/ComfyUI/custom_nodes/comfyui_controlnet_aux && \
	git clone https://github.com/shiimizu/ComfyUI-TiledDiffusion /content/ComfyUI/custom_nodes/ComfyUI-TiledDiffusion && \
	git clone https://github.com/WASasquatch/was-node-suite-comfyui /content/ComfyUI/custom_nodes/was-node-suite-comfyui && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/PitEyeDetailer-v2-seg.pt -d /content/ComfyUI/models/ultralytics/segm -o PitEyeDetailer-v2-seg.pt && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/4xRealWebPhoto_v4_dat2.safetensors -d /content/ComfyUI/models/upscale_models -o 4xRealWebPhoto_v4_dat2.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/dreamshaperXL_lightningDPMSDE.safetensors -d /content/ComfyUI/models/checkpoints -o dreamshaperXL_lightningDPMSDE.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/ultralytics/resolve/main/xinsir-controlnet-tile-sdxl-1.0.safetensors -d /content/ComfyUI/models/controlnet -o xinsir-controlnet-tile-sdxl-1.0.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py
