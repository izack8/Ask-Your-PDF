FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "python create_isaac_vector_store.py && streamlit run main.py --server.address 0.0.0.0 --server.port 8501"]