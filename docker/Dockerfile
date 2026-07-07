FROM nvcr.io/nvidia/isaac-lab:2.3.2

WORKDIR /workspace/crazyplayground 

COPY . . 

RUN /workspace/isaaclab/isaaclab.sh -p -m pip install -e "source/CrazyPlayGround[rl]"

ENTRYPOINT []
CMD ["bash"]