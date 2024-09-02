FROM python:3.10.14-bookworm

USER root

# set environment variable
ENV WS=/workspace \
    DEBIAN_FRONTEND=noninteractive \
    TIMEZONE=Asia/Taipei

# Copy file to /workspace
COPY core $WS/core
COPY tools $WS/tools
COPY app.py schema.py validator.py vectorization.py $WS/

# Setting TimeZone
RUN apt -yq update && \
    apt -yq install tzdata && \
    ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Install package
COPY [ "./docker/requirements.txt", "/tmp/requirements.txt" ]
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 && rm /tmp/requirements.txt

# Move to workspace
WORKDIR $WS

# Entry
CMD [ "bin/bash" ]