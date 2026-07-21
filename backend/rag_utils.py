import os
import pickle
import re
import json
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from google import genai
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


# ================= GEMINI SETUP =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# A model can be overridden in backend/.env (for example, GEMINI_MODEL=...).
# Do not use the obsolete ``models/`` prefix here.  The Gemini SDK expects a
# model ID, and tries newer models first while retaining stable fallbacks.
CHAT_MODELS = [
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
CHAT_MODELS = list(dict.fromkeys(model for model in CHAT_MODELS if model))


def clean_answer_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^\*+\s*", "", text)
    return text.replace("**", "")


def format_package_amount(amount: float) -> str:
    lakhs = amount / 100000
    formatted = f"{lakhs:.2f}".rstrip("0").rstrip(".")
    unit = "lakh" if formatted == "1" else "lakhs"
    return f"{formatted} {unit}"


def find_company_in_question(question_lower: str, docs: list) -> str | None:
    companies = {
        str(d.get("company", "")).strip()
        for d in docs
        if str(d.get("company", "")).strip()
    }

    for company in sorted(companies, key=len, reverse=True):
        company_lower = company.lower()
        company_tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", company_lower)
            if len(token) > 2
        ]

        if company_lower in question_lower:
            return company

        if any(re.search(rf"\b{re.escape(token)}\b", question_lower) for token in company_tokens):
            return company

    return None


# ================= SAFE GEMINI WRAPPER =================

def safe_generate(prompt: str) -> str:
    last_error = None

    for model in CHAT_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if hasattr(response, "text"):
                return response.text

            return str(response)

        except Exception as e:
            last_error = e
            print(f"REAL GEMINI ERROR ({model}):", str(e))

    # Callers use a local, data-based fallback instead of exposing provider
    # errors (for example a retired model) to students in the chat window.
    return ""

# ================= TF-IDF EMBEDDING =================

vectorizer = TfidfVectorizer()


def embed_texts(texts):
    vectors = vectorizer.fit_transform(texts)
    return vectors.toarray().astype("float32")


# ================= LOAD INDEX =================

def load_index(index_dir=None):
    if index_dir is None:
        index_dir = os.path.join(BASE_DIR, "index")

    index = faiss.read_index(
        os.path.join(index_dir, "faiss.index")
    )

    with open(
        os.path.join(index_dir, "docs.pkl"),
        "rb"
    ) as f:
        docs = pickle.load(f)

    return index, docs


# ================= RETRIEVAL =================

def retrieve_similar(query, faiss_index, docs, k=5):
    try:
        # Use same text corpus
        corpus = [doc["text"] for doc in docs]

        # Fit vectorizer on corpus
        vectorizer.fit(corpus)

        # Transform query
        q_vec = vectorizer.transform([query])

        q_vec = q_vec.toarray().astype("float32")

        # Search FAISS
        distances, indices = faiss_index.search(q_vec, k)

        print("Distances:", distances)
        print("Indices:", indices)

        results = []

        for rank, idx in enumerate(indices[0]):

            # Ignore invalid FAISS index
            if idx == -1:
                continue

            doc = docs[int(idx)].copy()

            doc["score"] = float(distances[0][rank])

            results.append(doc)

        return results

    except Exception as e:
        print("Retrieval Error:", str(e))
        return []
# ================= RAG PROMPT =================

def build_rag_prompt(
    context_docs: List[Dict],
    question: str
) -> str:

    context = "\n\n---\n\n".join(
        [
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ]
    )

    return f"""
You are a helpful assistant for the College Placement Cell.

Use ONLY the context below.

If answer is not found, say:
"Data not found in records."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def _placement_records(docs: list) -> list:
    return [doc for doc in docs if doc.get("doc_type") != "placement_package"]


def _packages(docs: list) -> list:
    records = []
    for doc in docs:
        if doc.get("doc_type") != "placement_package":
            continue
        try:
            records.append((float(doc.get("package_amount")), doc))
        except (TypeError, ValueError):
            continue
    return records


def _matches(doc: dict, year: str | None, department: str | None, company: str | None) -> bool:
    if year and str(doc.get("year", "")) != year:
        return False
    if department and str(doc.get("department", "")).upper() != department:
        return False
    return not company or str(doc.get("company", "")).strip().upper() == company.upper()


def _company_summary(company: str, docs: list, year: str | None = None) -> dict:
    placements = [doc for doc in _placement_records(docs) if _matches(doc, year, None, company)]
    packages = [(amount, doc) for amount, doc in _packages(docs) if _matches(doc, year, None, company)]
    years = sorted({str(doc.get("year", "")) for doc in placements if doc.get("year")})
    departments = sorted({str(doc.get("department", "")).upper() for doc in placements if doc.get("department")})
    return {
        "placements": placements,
        "packages": packages,
        "years": years,
        "departments": departments,
    }


def local_fallback_answer(question: str, docs: list) -> dict:
    """Return a truthful result from placement records when Gemini is unavailable."""
    companies = sorted({str(doc.get("company", "")).strip() for doc in docs if doc.get("company")})
    return {
        "answer": (
            "I could not generate a detailed explanation right now, but the placement "
            f"records contain {len(companies)} companies. Try asking for a company, year, "
            "department, number of students, highest package, lowest package, or top recruiter."
        ),
        "sources": [],
    }


# ================= RAG =================


def answer_with_rag(question: str, faiss_index, docs, k: int = 20):

    question_lower = question.lower()

    try:

        # ---------------- YEAR ----------------
        year = None
        for y in ["2023", "2024", "2025"]:
            if y in question_lower:
                year = y

        # ---------------- DEPARTMENT ----------------
        department = None
        company_filter = find_company_in_question(question_lower, docs)

        branches = [
            "cse", "eee", "ece",
            "it", "aeie", "me",
            "ce", "csbs"
        ]

        for branch in branches:
            if re.search(rf"\b{re.escape(branch)}\b", question_lower):
                department = branch.upper()

        # ============================
        # COMPANY COMPARISON (record-based, no model required)
        # ============================
        mentioned_companies = []
        for company in sorted({str(d.get("company", "")).strip() for d in docs if d.get("company")}, key=len, reverse=True):
            if company.lower() in question_lower and company not in mentioned_companies:
                mentioned_companies.append(company)

        if len(mentioned_companies) >= 2 and any(word in question_lower for word in ("compare", "comparison", "better", "versus", " vs ")):
            left, right = mentioned_companies[:2]
            left_data, right_data = _company_summary(left, docs, year), _company_summary(right, docs, year)

            def package_text(data):
                if not data["packages"]:
                    return "Not available"
                return format_package_amount(max(amount for amount, _ in data["packages"]))

            def list_text(values):
                return ", ".join(values) if values else "Not available"

            return {
                "answer": (
                    f"Here is a comparison from the placement records{f' for {year}' if year else ''}:\n\n"
                    f"| Metric | {left} | {right} |\n|---|---:|---:|\n"
                    f"| Recorded placements | {len(left_data['placements'])} | {len(right_data['placements'])} |\n"
                    f"| Highest recorded package | {package_text(left_data)} | {package_text(right_data)} |\n"
                    f"| Hiring departments | {list_text(left_data['departments'])} | {list_text(right_data['departments'])} |\n"
                    f"| Years in records | {list_text(left_data['years'])} | {list_text(right_data['years'])} |\n\n"
                    "The records can compare past placements and packages, but they do not contain role, location, work culture, or individual offer details needed to say which company is objectively better to join."
                ),
                "sources": (left_data["placements"] + right_data["placements"])[:10],
            }

        # ============================
        # TOTAL COMPANIES IN YEAR
        # ============================

        if (
            "how many companies" in question_lower
            or "total company" in question_lower
            or "total companies" in question_lower
        ):

            matched_companies = set()

            for d in docs:

                yr = str(d.get("year", ""))

                if year and yr != year:
                    continue

                company = str(d.get("company", "")).strip()

                if company:
                    matched_companies.add(company)

            company_list = sorted(list(matched_companies))

            return {
                "answer":
                f"There are {len(company_list)} companies in {year}:\n\n"
                + "\n".join(
                    [f"{i+1}. {c}" for i, c in enumerate(company_list)]
                ),

                "sources": []
            }

        # ============================
        # COUNT STUDENTS
        # ============================

        if (
            "number of students" in question_lower
            or "count" in question_lower
            or re.search(r"how many\s+(?:[a-z]+\s+)?students", question_lower)
            or "how many students" in question_lower
        ):

            matched_students = []

            for d in docs:
                if d.get("doc_type") == "placement_package":
                    continue

                dept = str(
                    d.get("department", "")
                ).upper()

                yr = str(
                    d.get("year", "")
                )

                if year and yr != year:
                    continue

                if department and dept != department:
                    continue

                if company_filter and str(d.get("company", "")).upper() != company_filter.upper():
                    continue

                matched_students.append(d)

            return {
                "answer":
                f"There are {len(matched_students)} students"
                f"{' from ' + department if department else ''}"
                f"{' placed at ' + company_filter if company_filter else ''}"
                f"{' in ' + year if year else ''}.",

                "sources": matched_students[:10]
            }

        # ============================
        # HIGHEST HIRING COMPANY
        # ============================

        if (
            "highest placement" in question_lower
            or "highest hiring" in question_lower
            or "most students" in question_lower
        ):

            company_count = {}

            for d in docs:
                if d.get("doc_type") == "placement_package":
                    continue

                yr = str(d.get("year", ""))

                if year and yr != year:
                    continue

                company = d.get("company", "")

                if not company:
                    continue

                company_count[company] = (
                    company_count.get(company, 0)
                    + 1
                )

            highest = max(
                company_count,
                key=company_count.get
            )

            count = company_count[highest]

            return {
                "answer":
                f"{highest} hired the highest "
                f"number of students in {year} "
                f"({count} students).",

                "sources": []
            }

        # ============================
        # HIGHEST PACKAGE
        # ============================

        if (
            "highest package" in question_lower
            or "maximum package" in question_lower
            or "max package" in question_lower
        ):
            package_docs = []

            for d in docs:
                if d.get("doc_type") != "placement_package":
                    continue

                yr = str(d.get("year", ""))
                if year and yr != year:
                    continue

                company = str(d.get("company", "")).strip()
                if company_filter and company.upper() != company_filter.upper():
                    continue

                try:
                    amount = float(d.get("package_amount", ""))
                except (TypeError, ValueError):
                    continue

                package_docs.append((amount, d))

            if not package_docs:
                return {
                    "answer": "Data not found in records.",
                    "sources": []
                }

            amount, doc = max(package_docs, key=lambda item: item[0])
            answer_year = f" in {year}" if year else ""
            answer_company = (
                f" for {company_filter}"
                if company_filter
                else ""
            )

            return {
                "answer": (
                    f"The highest package{answer_company}{answer_year} is "
                    f"{format_package_amount(amount)}, "
                    f"offered by {doc.get('company', '')}."
                ),
                "sources": [doc]
            }

        # ============================
        # LOWEST PACKAGE
        # ============================
        if any(phrase in question_lower for phrase in ("lowest package", "minimum package", "min package")):
            package_docs = [
                (amount, doc) for amount, doc in _packages(docs)
                if _matches(doc, year, None, company_filter)
            ]
            if not package_docs:
                return {"answer": "Data not found in records.", "sources": []}

            amount, doc = min(package_docs, key=lambda item: item[0])
            return {
                "answer": (
                    f"The lowest recorded package"
                    f"{' for ' + company_filter if company_filter else ''}"
                    f"{' in ' + year if year else ''} is {format_package_amount(amount)}, "
                    f"offered by {doc.get('company', '')}."
                ),
                "sources": [doc],
            }

        # ============================
        # NORMAL RAG
        # ============================

        retrieved_docs = retrieve_similar(
            question,
            faiss_index,
            docs,
            k=k
        )

        prompt = build_rag_prompt(
            retrieved_docs,
            question
        )

        answer_text = clean_answer_text(safe_generate(prompt))
        if not answer_text:
            return local_fallback_answer(question, docs)

        return {
            "answer": answer_text,
            "sources": retrieved_docs
        }

    except Exception as e:

        print("REAL RAG ERROR:", str(e))

        return {
            "answer": f"RAG Error: {str(e)}",
            "sources": []
        }
# ================= FEW SHOT =================

def answer_with_few_shot(question: str):

    few_shot_prompt = f"""
You are a general AI assistant.

You DO NOT have access to placement database.

If asked for exact college placement records,
reply:
"I do not have access to actual records."

Question:
{question}

Answer:
"""

    try:
        answer_text = clean_answer_text(safe_generate(
            few_shot_prompt
        ))

        return {
            "answer": answer_text,
            "sources": []
        }

    except Exception as e:
        print("Few Shot Error:", str(e))

        return {
            "answer":
            "⚠️ Internal server error during Few-Shot processing.",
            "sources": []
        }


# ================= ZERO SHOT =================

def answer_with_zero_shot(question: str):

    zero_shot_prompt = f"""
You are a helpful AI assistant.

Answer using general knowledge.

Follow the user's requested format:
- If they ask for a comparison, use a markdown table with clear columns and short cell text.
- If they ask for bullet points or points, use bullet points.
- If they ask for a table, return a clear markdown table.
- Do not use markdown bold markers like **.
- Avoid one long paragraph when a structured answer is requested.
- For comparisons, start with one short sentence, then the table, then 1-2 concise takeaway bullets.
- Keep table headers simple, such as Feature, Option 1, Option 2, Difference, or Best for.

If unsure, say:
"I am not sure."

Question:
{question}

Answer:
"""

    try:
        answer_text = clean_answer_text(safe_generate(
            zero_shot_prompt
        ))

        return {
            "answer": answer_text,
            "sources": []
        }

    except Exception as e:
        print("Zero Shot Error:", str(e))

        return {
            "answer":
            "⚠️ Internal server error during Zero-Shot processing.",
            "sources": []
        }
    
def _clean_tokens(text: str) -> set:
    stopwords = {
        "and", "the", "for", "with", "from", "that", "this", "have", "has",
        "are", "was", "were", "you", "your", "student", "students", "company",
        "department", "academic", "year", "name", "placed", "placement",
        "resume", "project", "college", "school", "email", "phone",
    }

    return {
        token
        for token in re.findall(r"[a-zA-Z0-9+#.]+", str(text).lower())
        if len(token) > 2 and token not in stopwords
    }


def _extract_list(value) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [item.strip() for item in re.split(r"[,|;/\s]+", str(value)) if item.strip()]


def extract_resume_profile(resume_text: str) -> dict:
    """
    Extract structured resume signals with Gemini, then fall back to token parsing.
    """
    fallback_tokens = sorted(_clean_tokens(resume_text))
    known_departments = {"CSE", "IT", "ECE", "EEE", "EE", "ME", "CE", "AEIE", "CSBS"}
    inferred_departments = [
        dept for dept in known_departments
        if re.search(rf"\b{re.escape(dept)}\b", resume_text, re.I)
    ]
    fallback_profile = {
        "technical_skills": fallback_tokens[:40],
        "domains": [],
        "projects": [],
        "departments": sorted(inferred_departments),
    }

    prompt = f"""
Extract a concise JSON object from this resume text.

Return only valid JSON with these keys:
technical_skills: array of programming languages, tools, frameworks, databases, CS concepts
domains: array of domains such as AI, web development, data analysis, networking, embedded, manufacturing
projects: array of short project/domain phrases
departments: array of likely academic branches such as CSE, IT, ECE, EEE, ME, CE, AEIE, CSBS

RESUME:
{resume_text[:12000]}
"""

    response = safe_generate(prompt)

    try:
        json_text = response.strip()
        if "```" in json_text:
            json_text = re.sub(r"^```(?:json)?|```$", "", json_text, flags=re.MULTILINE).strip()
        profile = json.loads(json_text)
    except Exception:
        return fallback_profile

    return {
        "technical_skills": _extract_list(profile.get("technical_skills")) or fallback_profile["technical_skills"],
        "domains": _extract_list(profile.get("domains")),
        "projects": _extract_list(profile.get("projects")),
        "departments": [item.upper() for item in _extract_list(profile.get("departments"))] or fallback_profile["departments"],
    }


def load_company_profiles(docs: list) -> list:
    profiles = []
    existing = set()

    for doc in docs:
        company = str(doc.get("company", "")).strip()
        if company and company.upper() not in existing:
            profiles.append({
                "company": company,
                "role": "Placement opportunity",
                "required_skills": [],
                "preferred_skills": [],
                "departments": [],
            })
            existing.add(company.upper())

    return profiles


def _company_history(company: str, docs: list) -> dict:
    years = set()
    departments = set()
    count = 0

    for doc in docs:
        if str(doc.get("company", "")).strip().upper() != company.upper():
            continue

        count += 1
        year = str(doc.get("year", "")).strip()
        department = str(doc.get("department", "")).strip().upper()
        if year:
            years.add(year)
        if department:
            departments.add(department)

        stream_match = re.search(r"(?:Stream|Department):\s*([^|]+)", str(doc.get("text", "")), re.I)
        if stream_match:
            departments.add(stream_match.group(1).strip().upper())

        year_match = re.search(r"(?:Academic Year|Year):\s*(\d{4})", str(doc.get("text", "")), re.I)
        if year_match:
            years.add(year_match.group(1))

    return {
        "years": sorted(years),
        "departments": sorted(departments),
        "placement_count": count,
    }


def _overlap(source_items: list, target_items: list) -> list:
    source_tokens = _clean_tokens(" ".join(source_items))
    target_tokens = _clean_tokens(" ".join(target_items))
    return sorted(source_tokens.intersection(target_tokens))


def explain_company_match(suggestion: dict) -> str:
    skills = ", ".join(suggestion.get("matched_required", [])[:4] + suggestion.get("matched_preferred", [])[:3])
    missing = ", ".join(suggestion.get("missing_required", [])[:3])

    if skills and missing:
        return (
            f"Good fit for {suggestion['role']} because your resume matches {skills}. "
            f"To improve fit, strengthen {missing}."
        )
    if skills:
        return f"Good fit for {suggestion['role']} because your resume matches {skills}."
    if suggestion.get("department_match"):
        return "Potential fit because your branch aligns with this company's historical hiring pattern."
    return "Potential fit based on this company's past placement activity."

def match_resume_to_companies(resume_text: str, docs: list) -> list:
    """
    Rank companies using extracted resume skills, company skill profiles, and
    historical placement records.
    """
    resume_profile = extract_resume_profile(resume_text)
    resume_skills = (
        resume_profile["technical_skills"]
        + resume_profile["domains"]
        + resume_profile["projects"]
    )
    resume_departments = {item.upper() for item in resume_profile.get("departments", [])}
    profiles = load_company_profiles(docs)

    recommendations = []

    for company_profile in profiles:
        history = _company_history(company_profile["company"], docs)
        profile_departments = set(company_profile.get("departments") or history["departments"])
        matched_required = _overlap(resume_skills, company_profile.get("required_skills", []))
        matched_preferred = _overlap(resume_skills, company_profile.get("preferred_skills", []))
        resume_skill_tokens = _clean_tokens(" ".join(resume_skills))
        missing_required = [
            skill for skill in company_profile.get("required_skills", [])
            if _clean_tokens(skill).isdisjoint(resume_skill_tokens)
        ]
        department_match = bool(
            resume_departments
            and profile_departments
            and resume_departments.intersection(profile_departments)
        )

        history_score = min(history["placement_count"], 25) * 0.4
        match_score = (
            len(matched_required) * 12
            + len(matched_preferred) * 6
            + (10 if department_match else 0)
            + history_score
        )

        if match_score <= 0:
            continue

        suggestion = {
            "company": company_profile["company"],
            "role": company_profile["role"],
            "match_score": round(match_score, 2),
            "matched_keywords": sorted(set(matched_required + matched_preferred))[:8],
            "matched_required": matched_required[:8],
            "matched_preferred": matched_preferred[:8],
            "missing_required": missing_required[:5],
            "years": history["years"],
            "departments": sorted(profile_departments or set(history["departments"]))[:6],
            "placement_count": history["placement_count"],
            "department_match": department_match,
        }
        suggestion["explanation"] = explain_company_match(suggestion)
        recommendations.append(suggestion)

    recommendations.sort(key=lambda item: item["match_score"], reverse=True)

    return recommendations[:5]
