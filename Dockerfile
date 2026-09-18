FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STORAGE_ROOT=/data
COPY requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml ./
COPY iiip_agent ./iiip_agent
RUN pip install --no-cache-dir --no-deps . && useradd --uid 10001 --create-home agent && mkdir /data && chown agent:agent /data
USER agent
EXPOSE 8000
CMD ["python", "-m", "iiip_agent"]
