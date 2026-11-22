# Shopping Sessions: native C++ pipeline.
# The build stage compiles one static binary with clang, -march=x86-64-v3
# (AVX2/BMI2). There is no AVX-512 build: on Cascade Lake class scorers 512-bit
# instructions lower the core clock, and even where they do not, the v4 build
# measured 1.7-3% more CPU time than v3. One binary also saves the exec of a
# second one at start.
FROM ubuntu:24.04 AS build
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends clang libstdc++-14-dev libc6-dev >/dev/null && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ /src/
ARG CXXFLAGS="-O3 -std=c++17 -static -pthread -fno-stack-protector"
RUN clang++ $CXXFLAGS -march=x86-64-v3 -o /pipeline main.cpp

# The binary is static, so the runtime image is empty apart from it
# (faster container start than a distribution base image).
FROM scratch
COPY --from=build /pipeline /app/
# glibc's memcpy/memset stay on 256-bit registers (this is already glibc's
# default on Cascade Lake and Ice Lake; the setting only pins it).
ENV GLIBC_TUNABLES=glibc.cpu.hwcaps=Prefer_No_AVX512
WORKDIR /app
CMD ["/app/pipeline"]
