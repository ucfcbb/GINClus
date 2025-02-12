FROM python:3.8

WORKDIR /ginclus_container

ENV VENV=/opt/venv
ENV PATH="$VENV/bin:$PATH"
RUN python3 -m venv $VENV && pip3 install --upgrade pip

WORKDIR /ginclus_container
ADD . /ginclus_container
RUN pip3 install -r requirements.txt --no-cache-dir
