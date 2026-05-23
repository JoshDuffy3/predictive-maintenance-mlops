FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python src/download_data.py && python src/train.py

EXPOSE 5000

CMD ["python", "app/app.py"]
