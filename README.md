# Redrob Candidate Ranker

Deterministic, CPU-only candidate ranker for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Reproduce

```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
python3 validate_submission.py ./submission.csv
```

The ranker uses only the Python standard library. It streams the JSONL input and keeps only the current top candidates in memory.

## Hugging Face Docker Space

This repository can run as a Hugging Face Docker Space. The Space starts a
minimal Gradio app on port 7860, accepts a `.json` or `.jsonl` candidate file,
runs the existing `rank.py`, and returns `submission.csv` for download.

Upload the project files to a new Hugging Face Space with Docker as the SDK,
then let Hugging Face build the included `Dockerfile`.

Required Space files:

- `app.py`
- `Dockerfile`
- `requirements.txt`
- `rank.py`
- `validate_submission.py`
- `README.md`
- `candidate_schema.json`
- `sample_candidates.json`
- Any other project documentation you want visible in the Space

Do not upload private or full challenge candidate datasets unless you intend
them to be public in the Space repository.

## Docker

The Docker image now starts the Gradio web app for Hugging Face Spaces:

```bash
docker build -t redrob-ranker:latest .
docker run --rm -p 7860:7860 redrob-ranker:latest
```

Then open `http://localhost:7860`.

Local command-line usage remains unchanged:

```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

To run the ranker directly inside the Docker image instead of starting Gradio:

```bash
docker build -t redrob-ranker:latest .
mkdir -p docker-output
docker run --rm \
  -v "$PWD/candidates.jsonl:/data/candidates.jsonl:ro" \
  -v "$PWD/docker-output:/out" \
  redrob-ranker:latest
python3 validate_submission.py docker-output/submission.csv
```

The Docker entrypoint also accepts the original ranker flags, so the existing
`docker-compose.yml` and `DOCKER.md` workflows continue to run `rank.py`.

## Approach

The job description is for a founding Senior AI Engineer focused on production retrieval, ranking, recommendation, embeddings, LLM integration, and evaluation systems. The scorer therefore avoids raw keyword counting as the main signal and combines:

- Role fit: current title and seniority, with strong weights for AI/ML/search/recommendation/NLP roles.
- Career evidence: production search, retrieval, ranking, recommender, vector database, evaluation, A/B testing, and scale indicators from work history.
- Trusted skills: relevant skills weighted by proficiency, duration, endorsements, and Redrob assessment scores.
- Product-company context: product-company experience is rewarded; services-only histories are down-weighted per the JD.
- Logistics and availability: India/Pune/Noida or relocation fit, recent activity, open-to-work, recent applications, response rate, response time, notice period, recruiter saves, interview completion, offer acceptance, and GitHub activity.
- Risk controls: penalties for keyword-stuffing profiles, inconsistent dates, unsupported expert skills, non-target titles with many AI skills but little career evidence, and CV/speech-heavy profiles without NLP/IR evidence.

The output score is the model's relative fit score after a monotonic sigmoid transform; it is useful for ordering but should not be read as a calibrated probability of hire. The output reasoning is generated from the same features used for ranking, so each row references facts that exist in the candidate profile and notes obvious concerns such as notice period or low recruiter response rate.

## 🚀 Live Demo

Hugging Face Space:
https://huggingface.co/spaces/MkSachdev/candidate-ranking-system
