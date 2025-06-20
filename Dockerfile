# Multi-stage build for Rust neural network server
FROM rust:1.87-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all source files
COPY . .

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false neural

# Copy binaries from builder stage
COPY --from=builder /app/target/release/neural_network /usr/local/bin/
COPY --from=builder /app/target/release/input_server /usr/local/bin/
COPY --from=builder /app/target/release/output_server /usr/local/bin/
COPY --from=builder /app/target/release/topology_tester /usr/local/bin/
COPY --from=builder /app/target/release/topology_monitor /usr/local/bin/

# Copy static files
COPY --from=builder /app/static /app/static

# Create directories for configs and data
RUN mkdir -p /app/config /app/data /app/logs && \
    chown -R neural:neural /app

# Switch to app user
USER neural

# Set working directory
WORKDIR /app

# Expose common ports
EXPOSE 8000 8001 8002 8080 12000 12001

# Default command (can be overridden)
CMD ["neural_network", "--help"]