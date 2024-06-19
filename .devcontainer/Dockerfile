FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
# && apt-get -y install --no-install-recommends <your-package-list-here>


RUN apt-get update && apt-get dist-upgrade -y && export DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y git

RUN conda install -c conda-forge jupyterlab ipywidgets

RUN pip install --no-cache-dir torch torchvision torchaudio

RUN pip install --no-cache-dir accelerate transformers bitsandbytes

RUN pip install --no-cache-dir langchain sentence-transformers pypdf

RUN conda install -c pytorch -c nvidia faiss-gpu=1.8.0