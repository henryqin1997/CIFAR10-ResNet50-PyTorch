docker pull nvcr.io/nvidia/pytorch:21.11-py3

echo runing docker container ${USER}_cifar

docker run -dit -shm-size 1G -v CIFAR10-ResNet50-PyTorch:/workspace/code --gpus all --name ${USER}_cifar nvcr.io/nvidia/pytorch:21.11-py3

docker exec -it ${USER}_cifar /bin/bash

cd code

python3 main.py --optimizer lars --max-lr 2.3
