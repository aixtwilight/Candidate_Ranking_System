FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY rank.py validate_submission.py requirements.txt README.md ./
COPY sample_candidates.json candidate_schema.json ./

RUN mkdir -p /data /out \
    && python -m compileall rank.py validate_submission.py

ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "/data/candidates.jsonl", "--out", "/out/submission.csv"]
