FROM python:3.10-slim

FROM python:3.10-slim

# Install system dependencies for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    python3-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Create non-root user
RUN useradd -m -u 1000 user

# Environment
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
RUN chown -R user:1000 /app
# Install pip and uv
RUN pip install --upgrade pip && pip install uv

# Copy project files
COPY --chown=user:1000 . /app

# Switch to non-root user
USER user

# Install dependencies
RUN uv sync

ENV HOST=0.0.0.0
ENV PORT=7860
# Run the server
CMD ["uv", "run", "/app/fastmcp_server.py"]

# Build the docker
# docker build -t codereview-mcp-server .
# run as
# docker run -it --rm -p 7860:7860 codereview-mcp-server