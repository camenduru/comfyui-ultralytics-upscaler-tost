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

RUN pip install opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 transformers==4.41.2 accelerate==0.30.1 insightface==0.7.3 onnxruntime==1.20.1 onnxruntime-gpu==1.20.1 color-matcher==0.5.0 pilgram==1.2.1 \
	ultralytics==8.3.49 segment-anything==1.0 piexif==1.1.3 qrcode==7.4.2 requirements-parser==0.9.0 rembg==2.0.57 rich==13.7.1 rich-argparse==1.5.1 matplotlib==3.8.4 pillow spandrel==0.3.4 \
	scikit-image==0.24.0 opencv-python-headless==4.10.0.84 GitPython==3.1.43 scipy==1.14.0 numpy==1.26.4 cachetools==5.4.0 librosa==0.10.2.post1 importlib-metadata==8.0.0 PyYAML==6.0.1 filelock==3.15.4 \
	mediapipe==0.10.14 svglib==1.5.1 fvcore==0.1.5.post20221221 yapf==0.40.2 omegaconf==2.3.0 ftfy==6.2.0 addict==2.4.0 yacs==0.1.8 albumentations==1.4.11 scikit-learn==1.5.1 fairscale==0.4.13 \
	git+https://github.com/WASasquatch/img2texture git+https://github.com/WASasquatch/cstr git+https://github.com/WASasquatch/ffmpy joblib==1.4.2 numba==0.60.0 timm==1.0.7 tqdm==4.66.4 kornia==0.7.4 && \
	git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /content/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack /content/ComfyUI/custom_nodes/ComfyUI-Impact-Subpack && \
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