FROM python:3.11-slim

WORKDIR /app

# Install Python deps first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (cached in image)
RUN python -m spacy download en_core_web_sm \
 && python -m spacy download es_core_news_sm


# Copy the rest of the app
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

