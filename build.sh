#!/bin/bash
# build.sh

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [BUILD_TYPE] [--clean] [--clang]"
    echo "  BUILD_TYPE: Debug or Release (default: Release)"
    echo "  --clean: Clean build directory before building"
    echo "  --clang: Use Clang compiler instead of GCC"
    exit 1
}

# Parse arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
USE_CLANG=false

for arg in "$@"; do
    case $arg in
        Debug|Release)
            BUILD_TYPE="$arg"
            ;;
        --clean)
            CLEAN_BUILD=true
            ;;
        --clang)
            USE_CLANG=true
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            echo "Error: Unknown argument '$arg'"
            show_usage
            ;;
    esac
done

echo "🚀 Starting APR build process in ${BUILD_TYPE} mode..."

# Handle git security in Docker
git config --global --add safe.directory /workspace/libAPR

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "🧹 Cleaning build directory..."
    if [ -d "build" ]; then
        rm -rf build/*
        echo "✅ Build directory cleaned"
    else
        echo "📁 Creating build directory..."
    fi
fi

# Create build directory if it doesn't exist
mkdir -p build
cd build

if [ "$BUILD_TYPE" = "Debug" ]; then
    CUDA_FLAGS="\
        -Xcompiler -fPIC \
        --generate-line-info \
        --compiler-options -g \
        -DCUDA_ERROR_CHECK \
        -D_DEBUG"
else
    CUDA_FLAGS="\
        -Xcompiler -fPIC \
        -O3 \
        --use_fast_math" 
fi

# Set compiler flags for Clang if requested
if [ "$USE_CLANG" = true ]; then
    echo "🔧 Using Clang compiler..."
    export CC=clang
    export CXX=clang++
    CMAKE_COMPILER_FLAGS="\
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++"
else
    CMAKE_COMPILER_FLAGS=""
fi

# Configure with CMake
echo "🔧 Configuring with CMake for ${BUILD_TYPE}..."
cmake \
    ${CMAKE_COMPILER_FLAGS} \
    -DAPR_USE_CUDA=OFF \
    -DAPR_BUILD_EXAMPLES=ON \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    ..

# Build
echo "🏗️ Building APR..."
make -j$(nproc)

echo "✅ Build complete!"

cd ..
cp build/examples/Example_get_apr . && \
echo "✅ Successfully copied Example_get_apr to current directory"
echo "🔍 Build type was: ${BUILD_TYPE}"
if [ "$USE_CLANG" = true ]; then
    echo "🔨 Compiler: Clang"
else
    echo "🔨 Compiler: GCC"
fi