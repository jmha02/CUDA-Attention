# CUDA-Attention
POSTECH CSED405: GPU Acc Computing / CUDA Implementation of Naive Attention &amp; Flash Attention
~~~
$ docker build -t simple-flash-attention .
$ docker run --gpus 1 -it --name flashattn_container simple-flash-attention /bin/bash
(docker) $ cd simple-flash-attention && python3 ./bench.py
~~~
