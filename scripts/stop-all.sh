#!/bin/bash
################################################################################
# Stop All Services Script
#
# Gracefully stops vLLM, Fish Speech, and Web Test Server.
# Uses SIGTERM first, then SIGKILL if processes don't stop.
#
# Usage: ./stop-all.sh [--force]
#
# Options:
#   --force    Skip graceful shutdown, immediately SIGKILL all processes
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_DIR="$REPO_ROOT/.pids"
VLLM_INSTALL_DIR="$REPO_ROOT/vllm-install"

# Default ports
VLLM_PORT=8000
FISH_PORT=8843
WEB_PORT=8844

# Parse arguments
FORCE_KILL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_KILL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f    Skip graceful shutdown, immediately kill all processes"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "  Stopping All Services"
echo "========================================================================"
echo ""

# Function to stop a process gracefully
stop_process() {
    local name=$1
    local pid_file=$2
    local port=$3
    
    local stopped=false
    
    # Try to get PID from file
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            log_info "Stopping $name (PID: $pid)..."
            
            if [ "$FORCE_KILL" = true ]; then
                kill -9 $pid 2>/dev/null || true
            else
                # Graceful shutdown with SIGTERM
                kill -15 $pid 2>/dev/null || true
                
                # Wait up to 10 seconds for graceful shutdown
                local wait=0
                while [ $wait -lt 10 ]; do
                    if ! ps -p $pid > /dev/null 2>&1; then
                        break
                    fi
                    sleep 1
                    wait=$((wait + 1))
                done
                
                # Force kill if still running
                if ps -p $pid > /dev/null 2>&1; then
                    log_warning "$name did not stop gracefully, forcing..."
                    kill -9 $pid 2>/dev/null || true
                fi
            fi
            
            stopped=true
        fi
        rm -f "$pid_file"
    fi
    
    # Also try to find and kill by port
    if [ -n "$port" ]; then
        local port_pids=$(lsof -t -i :$port 2>/dev/null || true)
        if [ -n "$port_pids" ]; then
            for pid in $port_pids; do
                if ps -p $pid > /dev/null 2>&1; then
                    log_info "Stopping process on port $port (PID: $pid)..."
                    if [ "$FORCE_KILL" = true ]; then
                        kill -9 $pid 2>/dev/null || true
                    else
                        kill -15 $pid 2>/dev/null || true
                        sleep 2
                        if ps -p $pid > /dev/null 2>&1; then
                            kill -9 $pid 2>/dev/null || true
                        fi
                    fi
                    stopped=true
                fi
            done
        fi
    fi
    
    if [ "$stopped" = true ]; then
        log_success "$name stopped"
    else
        log_info "$name was not running"
    fi
}

################################################################################
# 1. Stop Web Test Server (stop in reverse order)
################################################################################
stop_process "Web Test Server" "$PID_DIR/web-test.pid" $WEB_PORT

################################################################################
# 2. Stop Fish Speech Server
################################################################################
stop_process "Fish Speech" "$PID_DIR/fish-speech.pid" $FISH_PORT

################################################################################
# 3. Stop vLLM Server
################################################################################
# vLLM has its own PID file in the install directory
VLLM_PID_FILE="$VLLM_INSTALL_DIR/helpers/.vllm-server.pid"
stop_process "vLLM" "$VLLM_PID_FILE" $VLLM_PORT

# Also clean up the launcher PID if it exists
if [ -f "$PID_DIR/vllm-launcher.pid" ]; then
    LAUNCHER_PID=$(cat "$PID_DIR/vllm-launcher.pid")
    if ps -p $LAUNCHER_PID > /dev/null 2>&1; then
        kill -9 $LAUNCHER_PID 2>/dev/null || true
    fi
    rm -f "$PID_DIR/vllm-launcher.pid"
fi

################################################################################
# Cleanup any orphaned Python processes (optional, be careful)
################################################################################
# This is commented out by default to avoid killing unrelated processes
# Uncomment if you want aggressive cleanup:
#
# log_info "Checking for orphaned processes..."
# pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
# pkill -f "tools.api_server" 2>/dev/null || true
# pkill -f "web-test/server.py" 2>/dev/null || true

################################################################################
# Summary
################################################################################
echo ""
echo "========================================================================"
log_success "All services stopped"
echo "========================================================================"
echo ""

# Verify ports are free
check_port() {
    local port=$1
    local name=$2
    if lsof -i :$port > /dev/null 2>&1; then
        log_warning "Port $port ($name) still in use"
        return 1
    else
        log_info "Port $port ($name) is free"
        return 0
    fi
}

check_port $VLLM_PORT "vLLM"
check_port $FISH_PORT "Fish Speech"
check_port $WEB_PORT "Web Test"

echo ""
