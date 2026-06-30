#!/usr/bin/env python3
"""
Deterministic CPU-only ranker for the Redrob candidate ranking challenge.

The model is intentionally small and auditable: it streams candidates, extracts
job-specific evidence, applies availability/logistics modifiers, penalizes
honeypot-like inconsistencies, and writes the top 100 submission CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


REFERENCE_DATE = date(2026, 6, 1)
TOP_K = 100

SERVICES_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "mphasis",
}

PRODUCT_COMPANIES = {
    "pied piper",
    "hooli",
    "swiggy",
    "cred",
    "razorpay",
    "zomato",
    "flipkart",
    "meesho",
    "inmobi",
    "nykaa",
    "zoho",
    "freshworks",
    "vedantu",
    "ola",
    "paytm",
    "byju's",
    "upgrad",
    "policybazaar",
    "dream11",
    "pharmeasy",
    "phonepe",
    "unacademy",
    "genpact ai",
    "sarvam ai",
    "openai",
    "google",
    "microsoft",
    "amazon",
    "aws",
    "meta",
    "netflix",
    "uber",
    "linkedin",
    "atlassian",
    "stripe",
    "airbnb",
    "doordash",
    "databricks",
    "snowflake",
    "nvidia",
    "apple",
    "oracle",
    "salesforce",
    "adobe",
    "yellow.ai",
    "mad street den",
    "rephrase.ai",
    "glance",
    "wysa",
    "aganitha",
}

NON_TARGET_TITLES = {
    "business analyst",
    "hr manager",
    "mechanical engineer",
    "accountant",
    "project manager",
    "customer support",
    "operations manager",
    "content writer",
    "sales executive",
    "civil engineer",
    "graphic designer",
    "marketing manager",
}

TITLE_WEIGHTS = [
    (re.compile(r"\b(senior|staff|lead)\b.*\b(ai|ml|machine learning|applied scientist|nlp)\b"), 24.0),
    (re.compile(r"\b(recommendation systems|search)\s+engineer\b"), 23.0),
    (re.compile(r"\b(applied ml|machine learning|ml|ai|nlp)\s+engineer\b"), 21.0),
    (re.compile(r"\b(senior software engineer \(ml\)|senior data scientist)\b"), 18.0),
    (re.compile(r"\bdata scientist\b"), 15.0),
    (re.compile(r"\b(senior data engineer|data engineer|analytics engineer|backend engineer)\b"), 10.0),
    (re.compile(r"\bsoftware engineer|full stack developer|backend|cloud engineer|devops\b"), 7.0),
]

SKILL_WEIGHTS = {
    "information retrieval": 8.0,
    "semantic search": 8.0,
    "recommendation systems": 8.0,
    "vector search": 7.5,
    "embeddings": 7.0,
    "sentence transformers": 7.0,
    "faiss": 6.5,
    "pinecone": 6.0,
    "qdrant": 6.0,
    "milvus": 6.0,
    "opensearch": 5.8,
    "elasticsearch": 5.8,
    "rag": 5.5,
    "llms": 5.2,
    "fine-tuning llms": 4.8,
    "qlora": 4.4,
    "lora": 4.2,
    "hugging face transformers": 4.0,
    "transformers": 4.0,
    "pytorch": 3.8,
    "tensorflow": 3.2,
    "xgboost": 3.4,
    "lightgbm": 3.2,
    "learning to rank": 6.0,
    "bm25": 3.0,
    "mlops": 4.0,
    "kubeflow": 3.2,
    "bentoml": 3.0,
    "mlflow": 3.0,
    "feature engineering": 2.8,
    "data science": 2.6,
    "statistical modeling": 2.2,
    "spark": 2.0,
    "airflow": 1.8,
    "python": 2.0,
    "fastapi": 1.4,
    "kafka": 1.2,
    "data pipelines": 1.2,
}

CV_SPEECH_SKILLS = {
    "computer vision",
    "image classification",
    "object detection",
    "yolo",
    "opencv",
    "cnn",
    "gans",
    "diffusion models",
    "speech recognition",
    "asr",
    "tts",
}

PROFICIENCY_MULTIPLIER = {
    "beginner": 0.45,
    "intermediate": 0.75,
    "advanced": 1.0,
    "expert": 1.15,
}

EDUCATION_TIER_POINTS = {
    "tier_1": 4.0,
    "tier_2": 2.5,
    "tier_3": 1.0,
    "tier_4": 0.0,
    "unknown": 0.0,
}

CAREER_PATTERNS = [
    (re.compile(r"\b(search|retrieval|ranking|recommendation|recommender)\b"), 5.0),
    (re.compile(r"\b(learning[- ]to[- ]rank|ltr|xgboost|lightgbm|gradient[- ]boosted)\b"), 5.0),
    (re.compile(r"\b(embedding|embeddings|vector database|vector search|semantic search)\b"), 4.8),
    (re.compile(r"\b(bm25|faiss|pinecone|qdrant|milvus|weaviate|opensearch|elasticsearch)\b"), 4.2),
    (re.compile(r"\b(ndcg|mrr|map|offline benchmark|offline-online|evaluation framework|a/b test|ab test|relevance labeling)\b"), 4.4),
    (re.compile(r"\b(production|deployed|real users|scale|latency|index refresh|serving|queries per month)\b"), 2.8),
    (re.compile(r"\b(llm|llms|rag|fine[- ]tuning|lora|qlora|transformers)\b"), 2.4),
    (re.compile(r"\b(python|fastapi|backend|api|microservices)\b"), 1.8),
]

NEGATIVE_TEXT_PATTERNS = [
    (re.compile(r"\bpure research|academic lab|research-only\b"), 28.0),
    (re.compile(r"\blangchain tutorial|prompt engineering only|chatgpt\b"), 18.0),
    (re.compile(r"\bcomputer vision|opencv|robotics|speech recognition|asr|tts\b"), 2.2),
]


@dataclass(order=True)
class RankedCandidate:
    sort_score: float
    candidate_id: str
    score: float
    reasoning: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of ranked candidates to write")
    return parser.parse_args()


def open_candidates(path: str) -> Iterable[dict[str, Any]]:
    candidate_path = Path(path)
    opener = gzip.open if candidate_path.suffix == ".gz" else open
    with opener(candidate_path, "rt", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            for candidate in data:
                yield candidate
            return
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_log(value: float, cap: float) -> float:
    if value <= 0:
        return 0.0
    return clamp(math.log1p(value) / math.log1p(cap), 0.0, 1.0)


def text_blob(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_company", ""),
        profile.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.extend(
            [
                job.get("title", ""),
                job.get("company", ""),
                job.get("industry", ""),
                job.get("description", ""),
            ]
        )
    for edu in candidate.get("education", []):
        parts.extend([edu.get("degree", ""), edu.get("field_of_study", ""), edu.get("institution", "")])
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    return " ".join(str(part) for part in parts if part).lower()


def title_score(title: str) -> float:
    low = title.lower()
    for pattern, weight in TITLE_WEIGHTS:
        if pattern.search(low):
            return weight
    if low in NON_TARGET_TITLES:
        return -11.0
    return 0.0


def experience_score(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 14.0
    if 4.0 <= years < 5.0:
        return 10.5
    if 9.0 < years <= 11.0:
        return 10.0
    if 3.0 <= years < 4.0:
        return 5.0
    if 11.0 < years <= 14.0:
        return 4.0
    return -6.0


def career_score(candidate: dict[str, Any], low_text: str) -> tuple[float, list[str]]:
    score = 0.0
    evidence = []
    for pattern, weight in CAREER_PATTERNS:
        matches = len(pattern.findall(low_text))
        if matches:
            score += min(weight * matches, weight * 3.0)
            evidence.append(pattern.pattern)

    jobs = candidate.get("career_history", [])
    companies = [str(job.get("company", "")).lower() for job in jobs]
    if any(company in PRODUCT_COMPANIES for company in companies):
        score += 7.0
        evidence.append("product-company")
    if companies and all(company in SERVICES_COMPANIES for company in companies):
        score -= 8.0
        evidence.append("services-only")

    current = jobs[0] if jobs else {}
    if current.get("is_current") and int(current.get("duration_months") or 0) >= 18:
        score += 2.0
    if jobs:
        short_roles = sum(1 for job in jobs if int(job.get("duration_months") or 0) < 18)
        if len(jobs) >= 4 and short_roles / len(jobs) > 0.55:
            score -= 4.5
            evidence.append("job-hopping")

    for pattern, penalty in NEGATIVE_TEXT_PATTERNS:
        if pattern.search(low_text):
            score -= penalty
            evidence.append("negative-specialization")

    return score, evidence


def skill_score(candidate: dict[str, Any]) -> tuple[float, int, int, list[str]]:
    score = 0.0
    core_hits = 0
    cv_speech_hits = 0
    best_skills = []
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}

    for skill in candidate.get("skills", []):
        name = str(skill.get("name", "")).lower()
        base = SKILL_WEIGHTS.get(name, 0.0)
        if name in CV_SPEECH_SKILLS:
            cv_speech_hits += 1
        if base <= 0:
            continue

        core_hits += 1
        proficiency = PROFICIENCY_MULTIPLIER.get(skill.get("proficiency"), 0.65)
        duration = int(skill.get("duration_months") or 0)
        endorsements = int(skill.get("endorsements") or 0)
        duration_factor = 0.55 + 0.45 * clamp(duration / 36.0, 0.0, 1.0)
        endorsement_factor = 0.85 + 0.15 * clamp(endorsements / 40.0, 0.0, 1.0)
        assessment = assessments.get(skill.get("name"))
        assessment_factor = 1.0 if assessment is None else 0.85 + 0.30 * clamp(float(assessment) / 100.0, 0.0, 1.0)
        points = base * proficiency * duration_factor * endorsement_factor * assessment_factor
        score += points
        best_skills.append((points, skill.get("name", "")))

    best_skills.sort(reverse=True)
    if cv_speech_hits > core_hits:
        score -= min(8.0, (cv_speech_hits - core_hits) * 1.5)
    return min(score, 52.0), core_hits, cv_speech_hits, [name for _, name in best_skills[:4]]


def education_score(candidate: dict[str, Any]) -> float:
    score = 0.0
    relevant_fields = ("computer", "data science", "machine learning", "artificial intelligence", "information technology")
    for edu in candidate.get("education", []):
        score = max(score, EDUCATION_TIER_POINTS.get(edu.get("tier", "unknown"), 0.0))
        field = str(edu.get("field_of_study", "")).lower()
        if any(term in field for term in relevant_fields):
            score += 1.5
    return min(score, 5.5)


def location_score(profile: dict[str, Any], signals: dict[str, Any]) -> float:
    country = str(profile.get("country", "")).lower()
    location = str(profile.get("location", "")).lower()
    score = 0.0
    if country == "india":
        score += 4.0
    else:
        score -= 5.0
    if "pune" in location or "noida" in location:
        score += 4.0
    elif any(city in location for city in ("hyderabad", "mumbai", "delhi", "gurgaon", "bangalore")):
        score += 2.5
    elif signals.get("willing_to_relocate"):
        score += 1.5
    if signals.get("preferred_work_mode") in {"hybrid", "flexible", "onsite"}:
        score += 1.0
    return score


def behavior_score(signals: dict[str, Any]) -> float:
    score = 0.0
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = max(0, (REFERENCE_DATE - last_active).days)
        score += 10.0 * clamp(1.0 - days_inactive / 180.0, 0.0, 1.0)
    if signals.get("open_to_work_flag"):
        score += 4.0

    applications = int(signals.get("applications_submitted_30d") or 0)
    if applications >= 10:
        score += 3.5
    elif applications >= 5:
        score += 2.0
    elif applications >= 2:
        score += 0.8

    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    response_hours = float(signals.get("avg_response_time_hours") or 999.0)
    score += 7.0 * clamp(response_rate, 0.0, 1.0)
    score += 3.5 * clamp(1.0 - response_hours / 168.0, 0.0, 1.0)

    notice = int(signals.get("notice_period_days") or 180)
    if notice <= 30:
        score += 4.0
    elif notice <= 60:
        score += 1.5
    elif notice > 120:
        score -= 3.0

    score += 2.0 * normalized_log(signals.get("saved_by_recruiters_30d") or 0, 20)
    score += 1.5 * normalized_log(signals.get("profile_views_received_30d") or 0, 100)
    score += 1.5 * normalized_log(signals.get("search_appearance_30d") or 0, 250)
    score += 2.0 * clamp(float(signals.get("interview_completion_rate") or 0.0), 0.0, 1.0)

    offer_acceptance = float(signals.get("offer_acceptance_rate") if signals.get("offer_acceptance_rate") is not None else -1)
    if offer_acceptance >= 0:
        score += 4.0 * (offer_acceptance - 0.45)
        if offer_acceptance < 0.25:
            score -= 2.0

    github = float(signals.get("github_activity_score") if signals.get("github_activity_score") is not None else -1)
    if github >= 0:
        score += 3.0 * clamp(github / 100.0, 0.0, 1.0)
    else:
        score -= 1.0

    score += 0.7 if signals.get("verified_email") else -0.7
    score += 0.7 if signals.get("verified_phone") else -0.7
    score += 0.6 if signals.get("linkedin_connected") else 0.0
    return score


def honeypot_penalty(candidate: dict[str, Any], core_hits: int, career_points: float, low_text: str) -> tuple[float, list[str]]:
    penalty = 0.0
    flags = []
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    signup = parse_date(signals.get("signup_date"))
    active = parse_date(signals.get("last_active_date"))
    if signup and active and active < signup:
        penalty += 12.0
        flags.append("last_active_before_signup")

    impossible_skills = 0
    for skill in candidate.get("skills", []):
        duration = int(skill.get("duration_months") or 0)
        endorsements = int(skill.get("endorsements") or 0)
        if skill.get("proficiency") == "expert" and duration < 6:
            impossible_skills += 1
        if skill.get("proficiency") in {"advanced", "expert"} and duration == 0 and endorsements == 0:
            impossible_skills += 1
    if impossible_skills:
        penalty += min(12.0, impossible_skills * 3.0)
        flags.append("unsupported_expert_skills")

    title = str(profile.get("current_title", "")).lower()
    if title in NON_TARGET_TITLES and core_hits >= 7 and career_points < 15:
        penalty += 16.0
        flags.append("keyword_stuffing")

    if re.search(r"\b(pure research|academic lab|research-only)\b", low_text):
        penalty += 24.0
        flags.append("research_without_deployment")

    if re.search(r"\b(langchain tutorial|prompt engineering only|chatgpt)\b", low_text):
        penalty += 18.0
        flags.append("framework_demo_profile")

    if "we're building competence on the ml side" in low_text and core_hits >= 8:
        penalty += 5.0
        flags.append("transitioner_not_senior_ai")

    for edu in candidate.get("education", []):
        if int(edu.get("end_year") or 0) < int(edu.get("start_year") or 0):
            penalty += 8.0
            flags.append("education_dates")

    years = float(profile.get("years_of_experience") or 0.0)
    history_months = sum(int(job.get("duration_months") or 0) for job in candidate.get("career_history", []))
    if history_months and abs((history_months / 12.0) - years) > 6.0:
        penalty += 4.0
        flags.append("experience_mismatch")

    return penalty, flags


def score_candidate(candidate: dict[str, Any]) -> tuple[float, str]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    low_text = text_blob(candidate)

    title_points = title_score(str(profile.get("current_title", "")))
    exp_points = experience_score(float(profile.get("years_of_experience") or 0.0))
    career_points, career_evidence = career_score(candidate, low_text)
    skills_points, core_hits, cv_speech_hits, top_skills = skill_score(candidate)
    edu_points = education_score(candidate)
    loc_points = location_score(profile, signals)
    behavior_points = behavior_score(signals)
    penalty, flags = honeypot_penalty(candidate, core_hits, career_points, low_text)

    raw = (
        title_points
        + exp_points
        + career_points
        + skills_points
        + edu_points
        + loc_points
        + behavior_points
        - penalty
    )

    if title_points < 0 and career_points < 20:
        raw -= 12.0
    if core_hits == 0 and career_points < 14:
        raw -= 8.0
    if "services-only" in career_evidence and not any(company in low_text for company in PRODUCT_COMPANIES):
        raw -= 4.0

    # The raw evidence score can get high for the strongest candidates because
    # multiple independent signals agree. Center the sigmoid near the top-100
    # cutoff so the CSV score keeps visible separation instead of saturating.
    score = 1.0 / (1.0 + math.exp(-((raw - 160.0) / 30.0)))
    score = clamp(score, 0.0, 0.9999)
    reasoning = build_reasoning(candidate, score, title_points, career_points, skills_points, core_hits, top_skills, flags)
    return score, reasoning


def build_reasoning(
    candidate: dict[str, Any],
    score: float,
    title_points: float,
    career_points: float,
    skills_points: float,
    core_hits: int,
    top_skills: list[str],
    flags: list[str],
) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    title = profile.get("current_title", "Candidate")
    years = float(profile.get("years_of_experience") or 0.0)
    location = profile.get("location", "unknown location")
    response = float(signals.get("recruiter_response_rate") or 0.0)
    notice = int(signals.get("notice_period_days") or 0)
    active = signals.get("last_active_date", "unknown")
    applications = int(signals.get("applications_submitted_30d") or 0)
    offer_acceptance = float(signals.get("offer_acceptance_rate") if signals.get("offer_acceptance_rate") is not None else -1)

    skill_text = ", ".join(top_skills[:3]) if top_skills else f"{core_hits} relevant AI/search skills"
    if career_points >= 28:
        fit = "strong production search/retrieval evidence"
        jd_link = "matches the JD's need for deployed retrieval/ranking systems"
    elif career_points >= 18:
        fit = "some production ML/search evidence"
        jd_link = "partly aligns with the JD's search and evaluation ownership"
    elif title_points >= 18:
        fit = "strong target-role title but thinner explicit systems evidence"
        jd_link = "fits the AI-engineer seniority target but has less explicit ranking-system proof"
    else:
        fit = "adjacent background with partial JD fit"
        jd_link = "is more of an adjacent fit than the JD's ideal search/ranking profile"

    concern_parts = []
    if notice > 90:
        concern_parts.append(f"{notice}-day notice")
    if response < 0.25:
        concern_parts.append(f"low {response:.2f} recruiter response rate")
    if offer_acceptance >= 0 and offer_acceptance < 0.25:
        concern_parts.append(f"low {offer_acceptance:.2f} offer acceptance")
    if flags:
        concern_parts.append("profile consistency concerns")
    concern = f" Concern: {', '.join(concern_parts)}." if concern_parts else ""

    offer_text = "no prior offer history" if offer_acceptance < 0 else f"{offer_acceptance:.2f} offer acceptance"
    activity_text = f"{applications} recent applications, response {response:.2f}, last active {active}"

    variant = sum(ord(ch) for ch in candidate.get("candidate_id", "")) % 4
    if variant == 0:
        return (
            f"{title} with {years:.1f} yrs in {location}; {fit}, so this {jd_link}. "
            f"Top matching skills: {skill_text}. Hiring signals: {activity_text}, notice {notice} days, {offer_text}.{concern}"
        )
    if variant == 1:
        return (
            f"{years:.1f} yrs as a {title}; career evidence points to {fit}, with {skill_text} as the clearest skill match. "
            f"That {jd_link}; availability adds {applications} recent applications, {response:.2f} response rate, {notice}-day notice, {offer_text}.{concern}"
        )
    if variant == 2:
        return (
            f"Ranks highly because the profile shows {fit} for the search/retrieval-heavy AI Engineer JD. "
            f"{title} based in {location}; {skill_text} supports the hybrid retrieval/LLM stack, while {activity_text} and {notice}-day notice indicate reachability.{concern}"
        )
    return (
        f"{title}, {years:.1f} yrs, {location}: strongest match comes from {fit} plus {skill_text}. "
        f"The JD emphasizes production ranking/evaluation over keyword AI exposure; hiring-readiness signals are {activity_text}, {offer_text}, notice {notice} days.{concern}"
    )


def rank_candidates(candidate_path: str, top_k: int) -> list[RankedCandidate]:
    heap: list[RankedCandidate] = []
    for candidate in open_candidates(candidate_path):
        score, reasoning = score_candidate(candidate)
        candidate_id = candidate["candidate_id"]
        # Candidate ID secondary key is inverted for heap ordering so lexically
        # smaller IDs win ties after sorting descending below.
        item = RankedCandidate(score, candidate_id, score, reasoning)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        else:
            if item.sort_score > heap[0].sort_score or (
                item.sort_score == heap[0].sort_score and item.candidate_id < heap[0].candidate_id
            ):
                heapq.heapreplace(heap, item)

    return sorted(heap, key=lambda item: (-item.score, item.candidate_id))


def write_submission(ranked: list[RankedCandidate], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(ranked, start=1):
            # Preserve the model score in the CSV. The tiny rank-based offset
            # only prevents rounded ties from violating validator ordering.
            display_score = item.score
            display_score = max(0.0, min(0.9999, display_score - (rank - 1) * 0.000001))
            writer.writerow([item.candidate_id, rank, f"{display_score:.6f}", item.reasoning])


def main() -> None:
    args = parse_args()
    ranked = rank_candidates(args.candidates, args.top_k)
    write_submission(ranked, args.out)


if __name__ == "__main__":
    main()
