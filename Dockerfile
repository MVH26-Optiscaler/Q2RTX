# Toolchain image for building Quake II RTX on Linux.
#   docker build -t q2rtx-builder .
#
# Build in place (Release) — artifacts land in your working tree:
#   docker run --rm -v "$(pwd):/root/q2rtx" q2rtx-builder
#
# Debug build with an extra CMake arg:
#   docker run --rm -v "$(pwd):/root/q2rtx" q2rtx-builder Debug -DCONFIG_VKPT_ENABLE_IMAGE_DUMPS=ON
#
# Mirrors the Linux job in .github/workflows/build.yml:
# Steam Runtime "sniper" SDK, gcc-14/g++-14, Ninja, CMake 4.1.2.
FROM registry.gitlab.steamos.cloud/steamrt/sniper/sdk:latest

# The sniper SDK ships an older CMake; Q2RTX + submodules need a newer one. CI pins 4.1.2.
ARG CMAKE_VERSION=4.1.2
RUN mkdir -p /opt \
    && wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
    && tar xzf "cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -C /opt \
    && rm "cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
ENV PATH="/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}"

# The wrapper runs the container as the host user (--user) against the bind-mounted
# repo, so build artifacts stay host-owned. Trust any mounted repo so CMake's
# `git rev-parse` at configure time doesn't trip git's safe.directory guard.
RUN git config --system --add safe.directory '*'

COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /root/q2rtx
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
