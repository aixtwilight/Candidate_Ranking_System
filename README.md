# Redrob Candidate Ranker

Deterministic, CPU-only candidate ranker for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Reproduce

```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv
python3 validate_submission.py ./submission.csv
```

The ranker uses only the Python standard library. It streams the JSONL input and keeps only the current top candidates in memory.

## Docker

```bash
docker build -t redrob-ranker:latest .
mkdir -p docker-output
docker run --rm \
  -v "$PWD/candidates.jsonl:/data/candidates.jsonl:ro" \
  -v "$PWD/docker-output:/out" \
  redrob-ranker:latest
python3 validate_submission.py docker-output/submission.csv
```

See `DOCKER.md` for sample-file and Docker Compose commands.

## Approach

The job description is for a founding Senior AI Engineer focused on production retrieval, ranking, recommendation, embeddings, LLM integration, and evaluation systems. The scorer therefore avoids raw keyword counting as the main signal and combines:

- Role fit: current title and seniority, with strong weights for AI/ML/search/recommendation/NLP roles.
- Career evidence: production search, retrieval, ranking, recommender, vector database, evaluation, A/B testing, and scale indicators from work history.
- Trusted skills: relevant skills weighted by proficiency, duration, endorsements, and Redrob assessment scores.
- Product-company context: product-company experience is rewarded; services-only histories are down-weighted per the JD.
- Logistics and availability: India/Pune/Noida or relocation fit, recent activity, open-to-work, recent applications, response rate, response time, notice period, recruiter saves, interview completion, offer acceptance, and GitHub activity.
- Risk controls: penalties for keyword-stuffing profiles, inconsistent dates, unsupported expert skills, non-target titles with many AI skills but little career evidence, and CV/speech-heavy profiles without NLP/IR evidence.

The output score is the model's relative fit score after a monotonic sigmoid transform; it is useful for ordering but should not be read as a calibrated probability of hire. The output reasoning is generated from the same features used for ranking, so each row references facts that exist in the candidate profile and notes obvious concerns such as notice period or low recruiter response rate.
