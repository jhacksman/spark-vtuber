#!/bin/bash
################################################################################
# vLLM Installation Script for NVIDIA DGX Spark (Blackwell GB10)
# Version: 1.0.0
# Author: DGX Spark Community
# License: MIT
#
# This script automates the complete installation of vLLM on DGX Spark systems
# with Blackwell GB10 GPUs, including all necessary fixes and optimizations.
#
# Usage: ./install.sh [OPTIONS]
# Options:
#   --install-dir DIR    Installation directory (default: $PWD/vllm-install)
#   --vllm-version HASH  vLLM git commit (default: 66a168a19 - tested with Blackwell)
#   --python-version VER Python version (default: 3.12)
#   --skip-tests         Skip post-installation tests
#   --help               Show this help message
################################################################################

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
INSTALL_DIR="$PWD/vllm-install"
VLLM_VERSION="66a168a197ba214a5b70a74fa2e713c9eeb3251a"  # vLLM commit with Blackwell fixes
TRITON_VERSION="4caa0328bf8df64896dd5f6fb9df41b0eb2e750a"  # Triton commit that works with Blackwell
PYTHON_VERSION="3.12"
SKIP_TESTS=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

################################################################################
# Pre-flight Checks
################################################################################

preflight_checks() {
    print_header "Pre-flight System Checks"

    log_info "Checking system requirements..."

    # Check if running on ARM64
    ARCH=$(uname -m)
    if [[ "$ARCH" != "aarch64" ]] && [[ "$ARCH" != "arm64" ]]; then
        log_warning "This script is designed for ARM64 architecture (DGX Spark)"
        log_warning "Detected architecture: $ARCH"
    fi

    # Check for NVIDIA GPU
    if ! check_command nvidia-smi; then
        log_error "nvidia-smi not found. NVIDIA drivers required."
        exit 1
    fi

    # Check GPU type
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log_info "Detected GPU: $GPU_NAME"

    if [[ ! "$GPU_NAME" =~ "GB10" ]]; then
        log_warning "This script is optimized for NVIDIA GB10 (Blackwell)"
        log_warning "Your GPU: $GPU_NAME"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check CUDA
    if ! check_command nvcc; then
        # Check common CUDA install locations
        if [ -x "/usr/local/cuda/bin/nvcc" ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
            log_info "Found CUDA at /usr/local/cuda, added to PATH"
        else
            log_error "CUDA toolkit not found. Please install CUDA 13.0+"
            exit 1
        fi
    fi

    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    log_info "CUDA version: $CUDA_VERSION"

    # Check disk space (need ~50GB)
    AVAILABLE_SPACE=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ "$AVAILABLE_SPACE" -lt 50 ]]; then
        log_error "Insufficient disk space. Need at least 50GB, have ${AVAILABLE_SPACE}GB"
        exit 1
    fi

    # Check Python development headers
    PYTHON_INCLUDE="/usr/include/python${PYTHON_VERSION}"
    if [ ! -f "${PYTHON_INCLUDE}/Python.h" ]; then
        log_warning "Python development headers not found at ${PYTHON_INCLUDE}"
        log_info "Will install python3-dev and python${PYTHON_VERSION}-dev during system dependencies"
    else
        log_info "Python development headers: OK"
    fi

    log_success "Pre-flight checks passed!"
}

################################################################################
# Install System Dependencies for Triton Build
################################################################################

install_system_deps() {
    print_header "Step 1b/8: Installing System Dependencies for Triton"

    log_info "Installing LLVM, Clang, and build tools required for Triton..."
    
    # Check if we have sudo access
    if ! sudo -n true 2>/dev/null; then
        log_warning "sudo access required to install system dependencies"
        log_info "Please enter your password when prompted"
    fi

    # Install LLVM and Clang (required for Triton)
    sudo apt-get update
    sudo apt-get install -y \
        llvm-17 \
        clang-17 \
        lld-17 \
        libclang-17-dev \
        build-essential \
        cmake \
        ninja-build \
        git \
        python3-dev \
        python3.12-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        libbz2-dev

    # Set up LLVM alternatives so clang/llvm commands work
    if [ -f /usr/bin/clang-17 ]; then
        sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
        sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
        sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-17 100
        sudo update-alternatives --install /usr/bin/lld lld /usr/bin/lld-17 100
        log_success "LLVM 17 configured as default"
    fi

    # Verify installations
    log_info "Verifying installations..."
    clang --version || log_warning "clang not found after installation"
    cmake --version || log_warning "cmake not found after installation"
    ninja --version || log_warning "ninja not found after installation"

    log_success "System dependencies installed!"
}

################################################################################
# Install uv Package Manager
################################################################################

install_uv() {
    print_header "Step 1/8: Installing uv Package Manager"

    if check_command uv; then
        UV_VERSION=$(uv --version | awk '{print $2}')
        log_info "uv already installed: v$UV_VERSION"
    else
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        log_success "uv installed successfully"
    fi

    # Verify installation
    if ! check_command uv; then
        log_error "uv installation failed"
        exit 1
    fi
}

################################################################################
# Create Python Virtual Environment
################################################################################

create_venv() {
    print_header "Step 2/8: Creating Python Virtual Environment"

    VENV_DIR="$INSTALL_DIR/.vllm"

    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Remove and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            return
        fi
    fi

    log_info "Creating Python $PYTHON_VERSION virtual environment..."
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    uv venv .vllm --python "$PYTHON_VERSION"

    log_success "Virtual environment created at $VENV_DIR"
}

################################################################################
# Install PyTorch
################################################################################

install_pytorch() {
    print_header "Step 3/8: Installing PyTorch with CUDA 13.0"

    log_info "Installing PyTorch 2.9.0+cu130..."
    source "$INSTALL_DIR/.vllm/bin/activate"

    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

    # Verify PyTorch installation
    log_info "Verifying PyTorch installation..."
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

    log_success "PyTorch installed successfully"
}

################################################################################
# Clone and Build Triton
################################################################################

install_triton() {
    print_header "Step 4/8: Installing Triton from Main Branch"

    TRITON_DIR="$INSTALL_DIR/triton"

    if [ -d "$TRITON_DIR" ]; then
        log_info "Triton directory exists, updating..."
        cd "$TRITON_DIR"
        git fetch
    else
        log_info "Cloning Triton repository..."
        cd "$INSTALL_DIR"
        git clone https://github.com/triton-lang/triton.git
        cd triton
    fi

    log_info "Checking out Triton commit $TRITON_VERSION (tested with Blackwell)..."
    git checkout "$TRITON_VERSION"
    git submodule update --init --recursive

    log_info "Installing Triton build dependencies..."
    source "$INSTALL_DIR/.vllm/bin/activate"
    uv pip install pip cmake ninja pybind11

    log_info "Building Triton (this takes ~5 minutes)..."
    # Use python -m pip for better build error handling with Triton
    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
    python -m pip install --no-build-isolation -v . 2>&1 | tee "$INSTALL_DIR/triton-build.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "Triton build failed. See $INSTALL_DIR/triton-build.log for details"
        log_info "Attempting alternative installation method..."
        python setup.py install 2>&1 | tee "$INSTALL_DIR/triton-build-alt.log"
        if [ $? -ne 0 ]; then
            log_error "Triton installation failed. This is a known issue with Triton builds."
            log_error "Please see TROUBLESHOOTING.md for manual Triton installation instructions."
            exit 1
        fi
    fi

    # Verify Triton installation
    log_info "Verifying Triton installation..."
    python -c "import triton; print('Triton version:', triton.__version__)"

    log_success "Triton installed successfully"
}

################################################################################
# Install Additional Dependencies
################################################################################

install_dependencies() {
    print_header "Step 5/8: Installing Additional Dependencies"

    source "$INSTALL_DIR/.vllm/bin/activate"

    log_info "Installing vLLM runtime dependencies..."
    # These are required at runtime but not automatically installed when building from source
    # Core dependencies
    uv pip install psutil cloudpickle regex cachetools sentencepiece numpy requests tqdm py-cpuinfo
    # Tokenization and model loading
    uv pip install transformers tokenizers protobuf tiktoken gguf einops
    # API server dependencies
    uv pip install "fastapi[standard]" aiohttp openai pydantic prometheus_client prometheus-fastapi-instrumentator
    # Structured output and parsing
    uv pip install "lm-format-enforcer==0.11.3" "outlines_core==0.2.11" "diskcache==5.6.3" "lark==1.2.2" partial-json-parser
    # Serialization and communication
    uv pip install pyzmq msgspec blake3 pybase64 cbor2
    # Image/video processing
    uv pip install pillow opencv-python-headless "mistral_common[image,audio]"
    # Compression and utilities
    uv pip install "compressed-tensors==0.12.2" "depyf==0.20.0" watchfiles python-json-logger scipy ninja setproctitle
    # OpenAI compatibility
    uv pip install "openai-harmony>=0.0.3" "anthropic==0.71.0"
    # Additional utilities
    uv pip install typing_extensions filelock pyyaml

    log_info "Installing xgrammar, setuptools-scm, and apache-tvm-ffi..."
    uv pip install xgrammar setuptools-scm apache-tvm-ffi==0.1.0b15 --prerelease=allow

    log_success "Dependencies installed successfully"
}

################################################################################
# Clone vLLM
################################################################################

clone_vllm() {
    print_header "Step 6/8: Cloning vLLM Repository"

    VLLM_DIR="$INSTALL_DIR/vllm"

    if [ -d "$VLLM_DIR" ]; then
        log_warning "vLLM directory already exists at $VLLM_DIR"
        read -p "Remove and re-clone? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VLLM_DIR"
        else
            log_info "Using existing vLLM directory"
            cd "$VLLM_DIR"
            return
        fi
    fi

    log_info "Cloning vLLM $VLLM_VERSION..."
    cd "$INSTALL_DIR"
    git clone --recursive https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout "$VLLM_VERSION"
    git submodule update --init --recursive

    log_success "vLLM repository cloned"
}

################################################################################
# Apply Critical Fixes
################################################################################

apply_fixes() {
    print_header "Step 7/8: Applying Critical Fixes"

    cd "$INSTALL_DIR/vllm"

    # Fix 1: pyproject.toml license field
    log_info "Fixing pyproject.toml license field..."
    sed -i 's/^license = "Apache-2.0"$/license = {text = "Apache-2.0"}/' pyproject.toml
    sed -i '/^license-files = /d' pyproject.toml

    # Fix 2: CMakeLists.txt SM100/SM120 MOE kernels (check if already applied)
    if grep -q 'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f"' CMakeLists.txt; then
        log_info "CMakeLists.txt SM100/SM120 fix already applied"
    else
        log_info "Applying CMakeLists.txt SM100/SM120 fix..."
        # Fix for CUDA 13.0+ (sm_100, sm_120)
        sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f"/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f"/' CMakeLists.txt
        # Fix for older CUDA (sm_121a)
        sed -i 's/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a"/cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;12.1a"/' CMakeLists.txt
    fi

    # Fix 3: GPT-OSS Triton MOE kernels for Qwen3/gpt-oss support
    if [ -f "$SCRIPT_DIR/patches/gpt_oss_triton_moe.patch" ]; then
        log_info "Applying GPT-OSS Triton MOE kernel patch for Qwen3/gpt-oss support..."
        if patch --dry-run -p1 < "$SCRIPT_DIR/patches/gpt_oss_triton_moe.patch" > /dev/null 2>&1; then
            patch -p1 < "$SCRIPT_DIR/patches/gpt_oss_triton_moe.patch"
            log_success "GPT-OSS Triton MOE kernel patch applied"
        else
            log_warning "GPT-OSS Triton MOE kernel patch already applied or conflicts"
        fi
    else
        log_warning "GPT-OSS Triton MOE kernel patch not found (skipping)"
    fi

    # Configure use_existing_torch
    log_info "Configuring vLLM to use existing PyTorch..."
    python3 use_existing_torch.py

    log_success "All fixes applied successfully"
}

################################################################################
# Build and Install vLLM
################################################################################

build_vllm() {
    print_header "Step 8/8: Building vLLM (15-20 minutes)"

    cd "$INSTALL_DIR/vllm"
    source "$INSTALL_DIR/.vllm/bin/activate"

    # Set environment variables for Blackwell GPU (GB10 = SM 12.1)
    # IMPORTANT: Must include SM 10.0a (Hopper) because vLLM's MOE kernels reference
    # cutlass_moe_mm_sm100 symbols even on Blackwell. Without SM 10.0a, you get:
    # "undefined symbol: _Z20cutlass_moe_mm_sm100..."
    export TORCH_CUDA_ARCH_LIST="10.0a;12.1a"
    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
    export MAX_JOBS=$(nproc)
    
    log_info "Building for CUDA architectures: $TORCH_CUDA_ARCH_LIST"

    # Clean previous build artifacts to ensure fresh build with correct arch
    if [ -d "build" ] || [ -n "$(find vllm -name '*.so' 2>/dev/null)" ]; then
        log_info "Cleaning previous build artifacts..."
        rm -rf build/
        find vllm -name "*.so" -delete 2>/dev/null || true
        log_success "Previous build artifacts cleaned"
    fi

    # Disable flashinfer-python - it has license field format issues on ARM64/aarch64
    # that cause pyproject.toml validation to fail. vLLM will use fallback attention.
    log_info "Disabling flashinfer-python (has license format issues on ARM64)..."
    if [ -f "requirements/cuda.txt" ]; then
        sed -i 's/^flashinfer-python/#flashinfer-python/' requirements/cuda.txt
        log_success "flashinfer-python disabled in requirements/cuda.txt"
    fi

    log_info "Starting vLLM build..."
    log_warning "This will take 15-20 minutes. Go grab a coffee!"

    # Build vLLM with pip (more reliable than uv for complex builds)
    # The -v flag ensures we can see what CUDA arch is being used
    pip install -e . --no-build-isolation -v 2>&1 | tee "$INSTALL_DIR/vllm-build.log"
    BUILD_STATUS=${PIPESTATUS[0]}

    if [ $BUILD_STATUS -ne 0 ]; then
        log_error "vLLM build failed. See $INSTALL_DIR/vllm-build.log for details"
        log_error "Last 50 lines of build log:"
        tail -50 "$INSTALL_DIR/vllm-build.log"
        exit 1
    fi

    # Verify the build produced .so files
    log_info "Verifying vLLM C extensions were built..."
    SO_COUNT=$(find vllm -name "*.so" 2>/dev/null | wc -l)
    if [ "$SO_COUNT" -eq 0 ]; then
        log_error "vLLM build completed but no .so files were produced!"
        log_error "This means the C extensions were not compiled."
        exit 1
    fi
    log_success "Found $SO_COUNT compiled extension files"

    log_success "vLLM built successfully!"
}

################################################################################
# Create Helper Scripts
################################################################################

create_helper_scripts() {
    print_header "Creating Helper Scripts"

    # Create environment activation script
    log_info "Creating vllm_env.sh..."
    cat > "$INSTALL_DIR/vllm_env.sh" << EOF
#!/bin/bash
# vLLM Environment Configuration for DGX Spark
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/.vllm/bin/activate"
export TORCH_CUDA_ARCH_LIST="10.0a;12.1a"
CUDA_PATH=\$(ls -d /usr/local/cuda* 2>/dev/null | head -1)
export TRITON_PTXAS_PATH="\$CUDA_PATH/bin/ptxas"
export PATH="\$CUDA_PATH/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"
# Cache tiktoken encodings to avoid re-downloading
export TIKTOKEN_CACHE_DIR="\$SCRIPT_DIR/.tiktoken_cache"
mkdir -p "\$TIKTOKEN_CACHE_DIR"
echo "=== vLLM Environment Active ==="
echo "Virtual env: \$VIRTUAL_ENV"
echo "CUDA arch: \$TORCH_CUDA_ARCH_LIST"
echo "Python: \$(which python)"
echo "==============================="
EOF
    chmod +x "$INSTALL_DIR/vllm_env.sh"

    # Copy helper scripts from repository
    if [ -d "$SCRIPT_DIR/helpers" ]; then
        log_info "Copying helper scripts..."
        cp "$SCRIPT_DIR/helpers/vllm-serve.sh" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/helpers/vllm-stop.sh" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/helpers/vllm-status.sh" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR"/vllm-*.sh
    fi

    log_success "Helper scripts created"
}

################################################################################
# Post-Installation Tests
################################################################################

run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_info "Skipping post-installation tests"
        return
    fi

    print_header "Post-Installation Tests"

    source "$INSTALL_DIR/vllm_env.sh"

    log_info "Test 1: Import vLLM..."
    python -c "import vllm; print('vLLM version:', vllm.__version__)"

    log_info "Test 2: Check CUDA availability..."
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA available')"

    log_info "Test 3: Check GPU detection..."
    python -c "import torch; print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0))"

    log_success "All tests passed!"
}

################################################################################
# Parse Command Line Arguments
################################################################################

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --vllm-version)
                VLLM_VERSION="$2"
                shift 2
                ;;
            --python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                head -20 "$0" | grep "^#" | sed 's/^# //'
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                log_info "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    parse_args "$@"

    # Convert INSTALL_DIR to absolute path to avoid issues after cd
    INSTALL_DIR="$(cd "$(dirname "$INSTALL_DIR")" 2>/dev/null && pwd)/$(basename "$INSTALL_DIR")"
    # Handle case where parent directory doesn't exist yet
    if [[ "$INSTALL_DIR" == "/$(basename "$INSTALL_DIR")" ]]; then
        INSTALL_DIR="$PWD/$(basename "$INSTALL_DIR")"
    fi

    print_header "vLLM Installation for DGX Spark (Blackwell GB10)"
    log_info "Installation directory: $INSTALL_DIR"
    log_info "vLLM version: $VLLM_VERSION"
    log_info "Python version: $PYTHON_VERSION"
    echo ""

    preflight_checks
    install_system_deps
    install_uv
    create_venv
    install_pytorch
    install_triton
    install_dependencies
    clone_vllm
    apply_fixes
    build_vllm
    create_helper_scripts
    run_tests

    print_header "Installation Complete!"
    echo ""
    log_success "vLLM has been successfully installed!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Activate the environment:"
    echo -e "   ${BLUE}source $INSTALL_DIR/vllm_env.sh${NC}"
    echo ""
    echo "2. Start vLLM server:"
    echo -e "   ${BLUE}cd $INSTALL_DIR${NC}"
    echo -e "   ${BLUE}./vllm-serve.sh${NC}"
    echo ""
    echo "3. Test the API:"
    echo -e "   ${BLUE}curl http://localhost:8000/v1/models${NC}"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Run main function
main "$@"
