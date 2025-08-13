FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python create_vector_store.py

EXPOSE 8080

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8080"]