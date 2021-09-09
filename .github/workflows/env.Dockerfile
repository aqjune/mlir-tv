FROM ubuntu:20.04 AS layers
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies
# copying .deb files into image and installing it doesn't work for some reason
RUN apt update && \
    apt install -y git g++ python3.9 python3.9-venv cmake ninja-build \
    default-jdk m4 libncurses5-dev
RUN python3.9 -m venv /venv --without-pip

# copy z3, cvc5, llvm
COPY opt /opt

# flatten layers and export
FROM scratch
LABEL author="mlir-tv team"
COPY --from=layers / /
CMD [ "/bin/bash" ]
