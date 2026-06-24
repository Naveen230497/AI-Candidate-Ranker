FROM python:3.10-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and data
COPY api.py .
COPY rank.py .
COPY jd.txt .
COPY src/ ./src/
COPY data/ ./data/

# Run the API server
CMD ["python", "api.py"]
