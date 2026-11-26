# Support Desk Agents — API container.
# Build:  docker build -t support-desk .
# Run:    docker run -p 8000:8000 support-desk
# With a real key: docker run -p 8000:8000 -e LLM_API_KEY=... support-desk
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached unless requirements change).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the code.
COPY . .

EXPOSE 8000

# The API runs in mock mode unless LLM_API_KEY is passed with -e.
# ${PORT:-8000}: hosts like Render assign the port at runtime.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
