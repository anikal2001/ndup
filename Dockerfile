# Application image. Model weights are baked only when models/ is present in the
# build context. HF_HUB_OFFLINE stops a missing snapshot from downloading at runtime.
# The cuda tag uses the same source with the [gpu] extra; PR CI does not build it.

FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/models

WORKDIR /app
COPY pyproject.toml README.md uv.lock ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models /models

ENTRYPOINT ["ndup"]
CMD ["--help"]
