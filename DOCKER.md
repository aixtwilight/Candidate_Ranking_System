# Docker Usage

Build the image:

```bash
docker build -t redrob-ranker:latest .
```

Run the full 100k candidate ranking by mounting the candidate pool and an output directory:

```bash
mkdir -p docker-output
docker run --rm \
  -v "$PWD/candidates.jsonl:/data/candidates.jsonl:ro" \
  -v "$PWD/docker-output:/out" \
  redrob-ranker:latest
```

Validate the generated CSV:

```bash
python3 validate_submission.py docker-output/submission.csv
```

Run against the bundled sample file inside the image:

```bash
docker run --rm \
  -v "$PWD/docker-output:/out" \
  redrob-ranker:latest \
  --candidates /app/sample_candidates.json \
  --out /out/sample_submission.csv \
  --top-k 10
```

Docker Compose alternative:

```bash
mkdir -p docker-output
docker compose up --build
python3 validate_submission.py docker-output/submission.csv
```

For a hosted Docker sandbox, push this image to a public registry and use the same volume contract:

```bash
docker run --rm -v /path/to/candidates.jsonl:/data/candidates.jsonl:ro -v /path/to/out:/out YOUR_IMAGE
```

