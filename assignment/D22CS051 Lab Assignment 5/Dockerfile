# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/project:/workspace/project --rm nvcr.io/nvidia/pytorch:22.07-py3
FROM nvcr.io/nvidia/pytorch:23.02-py3
WORKDIR /dlops
COPY ./*.py .
COPY ./requirements.txt .
RUN pip3 install -U scikit-learn
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]