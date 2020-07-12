# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM tanabeken/public:base.ubuntu18.04.amd64.20200712

# ==================================================================
# PointPillars
# ------------------------------------------------------------------

WORKDIR /root
RUN git clone https://github.com/kentanabe/SparseConvNet.git && \
    cd ./SparseConvNet && \
    git pull && \
    git checkout a0b9bbbbee335eeb98aba55377b8835bfde1fb26 && \
    python3 setup.py install && \
    cd .. && \
    rm -rf SparseConvNet && \
    ls
RUN git clone https://github.com/kentanabe/second.pytorch.git && \
    cd second.pytorch && \
    git pull && \
    cd ..
ENV PYTHONPATH=/root/second.pytorch
 
VOLUME ["/root/data"]
VOLUME ["/root/model"]
WORKDIR /root/second.pytorch/second

ENTRYPOINT ["bash"]
