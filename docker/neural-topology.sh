#!/bin/bash

# Neural Network Topology Testing Script
# Manages Docker-based neural network topologies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
TOPOLOGY="linear"
ACTION="start"
DURATION=60
RATE=1.0
PATTERN="random"
MONITOR_PORT=3000

print_usage() {
    echo "Neural Network Topology Testing Tool"
    echo ""
    echo "Usage: $0 [OPTIONS] ACTION"
    echo ""
    echo "Actions:"
    echo "  start TOPOLOGY    Start a topology (linear, star, mesh, ring)"
    echo "  stop TOPOLOGY     Stop a topology"
    echo "  test TOPOLOGY     Run tests on a topology"
    echo "  monitor          Start topology monitor"
    echo "  build            Build Docker images"
    echo "  clean            Clean up all containers and images"
    echo "  status           Show status of all containers"
    echo "  logs TOPOLOGY    Show logs for a topology"
    echo ""
    echo "Options:"
    echo "  -d, --duration SECONDS    Test duration (default: 60)"
    echo "  -r, --rate HZ            Data injection rate (default: 1.0)"
    echo "  -p, --pattern PATTERN    Data pattern (random, sine, step, pulse, xor)"
    echo "  -m, --monitor-port PORT  Monitor web interface port (default: 3000)"
    echo "  -v, --verbose            Verbose output"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start linear          # Start linear topology"
    echo "  $0 test star -d 120      # Test star topology for 2 minutes"
    echo "  $0 monitor -m 8080       # Start monitor on port 8080"
    echo "  $0 clean                 # Clean up everything"
}

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
}

check_docker_compose() {
    if ! docker compose version &> /dev/null; then
        error "Docker Compose is not available"
        exit 1
    fi
}

build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    log "Building main neural network image..."
    sudo docker build -t neural-network:latest -f Dockerfile .
    
    log "Building neural network server image..."
    sudo docker build -t neural-network-server:latest -f Dockerfile.server .
    
    success "Docker images built successfully"
}

start_topology() {
    local topology=$1
    local compose_file="$SCRIPT_DIR/topologies/${topology}.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        error "Topology file not found: $compose_file"
        exit 1
    fi
    
    log "Starting $topology topology..."
    
    cd "$SCRIPT_DIR"
    sudo docker compose -f "topologies/${topology}.yml" up -d
    
    success "$topology topology started"
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 10
    
    # Show status
    show_topology_status "$topology"
}

stop_topology() {
    local topology=$1
    local compose_file="$SCRIPT_DIR/topologies/${topology}.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        error "Topology file not found: $compose_file"
        exit 1
    fi
    
    log "Stopping $topology topology..."
    
    cd "$SCRIPT_DIR"
    sudo docker compose -f "topologies/${topology}.yml" down
    
    success "$topology topology stopped"
}

test_topology() {
    local topology=$1
    
    log "Testing $topology topology..."
    log "Duration: ${DURATION}s, Rate: ${RATE}Hz, Pattern: $PATTERN"
    
    # Check if topology is running
    if ! check_topology_running "$topology"; then
        warning "Topology $topology is not running. Starting it first..."
        start_topology "$topology"
    fi
    
    # Build the topology tester if needed
    cd "$PROJECT_ROOT"
    if [[ ! -f "target/release/topology_tester" ]]; then
        log "Building topology tester..."
        cargo build --release --bin topology_tester
    fi
    
    # Run the test
    log "Starting topology test..."
    ./target/release/topology_tester \
        --topology "$topology" \
        --duration "$DURATION" \
        --rate "$RATE" \
        --pattern "$PATTERN" \
        ${VERBOSE:+--verbose}
    
    success "Topology test completed"
}

start_monitor() {
    log "Starting topology monitor on port $MONITOR_PORT..."
    
    cd "$PROJECT_ROOT"
    
    # Build the monitor if needed
    if [[ ! -f "target/release/topology_monitor" ]]; then
        log "Building topology monitor..."
        cargo build --release --bin topology_monitor
    fi
    
    # Start the monitor
    log "Monitor will be available at: http://localhost:$MONITOR_PORT"
    ./target/release/topology_monitor --port "$MONITOR_PORT" &
    
    local monitor_pid=$!
    echo "$monitor_pid" > /tmp/neural_monitor.pid
    
    success "Topology monitor started (PID: $monitor_pid)"
    info "Open http://localhost:$MONITOR_PORT in your browser"
}

stop_monitor() {
    if [[ -f /tmp/neural_monitor.pid ]]; then
        local pid=$(cat /tmp/neural_monitor.pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm -f /tmp/neural_monitor.pid
            success "Topology monitor stopped"
        else
            warning "Monitor process not found"
            rm -f /tmp/neural_monitor.pid
        fi
    else
        warning "Monitor PID file not found"
    fi
}

check_topology_running() {
    local topology=$1
    local compose_file="$SCRIPT_DIR/topologies/${topology}.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    local running_containers=$(sudo docker compose -f "topologies/${topology}.yml" ps -q)
    
    if [[ -n "$running_containers" ]]; then
        return 0
    else
        return 1
    fi
}

show_topology_status() {
    local topology=$1
    local compose_file="$SCRIPT_DIR/topologies/${topology}.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        error "Topology file not found: $compose_file"
        return 1
    fi
    
    log "Status for $topology topology:"
    
    cd "$SCRIPT_DIR"
    sudo docker compose -f "topologies/${topology}.yml" ps
}

show_all_status() {
    log "Neural Network Container Status:"
    echo ""
    
    # Show all neural network related containers
    sudo docker ps -a --filter "name=neural" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    log "Docker Images:"
    sudo docker images --filter "reference=neural*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
}

show_logs() {
    local topology=$1
    local compose_file="$SCRIPT_DIR/topologies/${topology}.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        error "Topology file not found: $compose_file"
        exit 1
    fi
    
    log "Showing logs for $topology topology..."
    
    cd "$SCRIPT_DIR"
    sudo docker compose -f "topologies/${topology}.yml" logs -f
}

clean_all() {
    log "Cleaning up all neural network containers and images..."
    
    # Stop monitor if running
    stop_monitor
    
    # Stop all topologies
    for topology_file in "$SCRIPT_DIR/topologies"/*.yml; do
        if [[ -f "$topology_file" ]]; then
            local topology=$(basename "$topology_file" .yml)
            log "Stopping $topology topology..."
            cd "$SCRIPT_DIR"
            sudo docker compose -f "topologies/${topology}.yml" down 2>/dev/null || true
        fi
    done
    
    # Remove containers
    log "Removing neural network containers..."
    sudo docker ps -aq --filter "name=neural" | xargs -r sudo docker rm -f
    
    # Remove images
    log "Removing neural network images..."
    sudo docker images -q --filter "reference=neural*" | xargs -r sudo docker rmi -f
    
    # Clean up networks
    log "Cleaning up networks..."
    sudo docker network ls --filter "name=neural" -q | xargs -r sudo docker network rm 2>/dev/null || true
    
    success "Cleanup completed"
}

# Parse command line arguments
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -r|--rate)
            RATE="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -m|--monitor-port)
            MONITOR_PORT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        start|stop|test|monitor|build|clean|status|logs)
            ACTION="$1"
            if [[ "$ACTION" != "monitor" && "$ACTION" != "build" && "$ACTION" != "clean" && "$ACTION" != "status" ]]; then
                TOPOLOGY="$2"
                shift 2
            else
                shift
            fi
            ;;
        *)
            error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo -e "${PURPLE}"
    echo "🧠 Neural Network Topology Testing Tool"
    echo "========================================"
    echo -e "${NC}"
    
    check_docker
    check_docker_compose
    
    case "$ACTION" in
        start)
            if [[ -z "$TOPOLOGY" ]]; then
                error "Topology name required for start action"
                exit 1
            fi
            start_topology "$TOPOLOGY"
            ;;
        stop)
            if [[ -z "$TOPOLOGY" ]]; then
                error "Topology name required for stop action"
                exit 1
            fi
            stop_topology "$TOPOLOGY"
            ;;
        test)
            if [[ -z "$TOPOLOGY" ]]; then
                error "Topology name required for test action"
                exit 1
            fi
            test_topology "$TOPOLOGY"
            ;;
        monitor)
            start_monitor
            ;;
        build)
            build_images
            ;;
        clean)
            clean_all
            ;;
        status)
            show_all_status
            ;;
        logs)
            if [[ -z "$TOPOLOGY" ]]; then
                error "Topology name required for logs action"
                exit 1
            fi
            show_logs "$TOPOLOGY"
            ;;
        *)
            error "Unknown action: $ACTION"
            print_usage
            exit 1
            ;;
    esac
}

main