# Shopping Sessions: native C++ pipeline.
# The build stage compiles two static binaries from the same source with clang
# (about 7% faster than g++ 14 on this code):
#   /app/pipeline     -march=x86-64-v3 (AVX2/BMI2), runs on any Cascade Lake or newer
#   /app/pipeline-v4  -march=x86-64-v4 (AVX-512)
# /app/pipeline checks the CPU at start and execs pipeline-v4 only when every
# AVX-512 feature v4 needs is present, so a CPU without them never sees an
# AVX-512 instruction.
FROM ubuntu:24.04 AS build
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends clang libstdc++-14-dev libc6-dev >/dev/null && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ /src/
ARG CXXFLAGS="-O3 -std=c++17 -static -pthread -fno-stack-protector"
RUN clang++ $CXXFLAGS -march=x86-64-v4 -o /pipeline-v4 main.cpp && \
    clang++ $CXXFLAGS -march=x86-64-v3 -DSS_DISPATCH_TO='"/app/pipeline-v4"' -o /pipeline main.cpp

# The binaries are static, so the runtime image is empty apart from them
# (faster container start than a distribution base image).
FROM scratch
COPY --from=build /pipeline /pipeline-v4 /app/
WORKDIR /app
CMD ["/app/pipeline"]
