import io
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


APP_TITLE = "Professional Resume–Job Matcher"

SKILL_ALIASES: Dict[str, List[str]] = {
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript"],
    "C++": ["c++"],
    "C#": ["c#", "c sharp"],
    "Go": ["golang", "go language"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo db"],
    "Oracle": ["oracle"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular"],
    "Vue": ["vue", "vue.js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Spring Boot": ["spring boot"],
    "REST APIs": ["rest api", "restful api", "rest services"],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "micro services"],
    "HTML": ["html"],
    "CSS": ["css"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "NLP": ["natural language processing", "nlp"],
    "Computer Vision": ["computer vision"],
    "Generative AI": ["generative ai", "genai", "gen ai"],
    "LLMs": ["large language models", "llm", "llms"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud platform", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "Jenkins": ["jenkins"],
    "Git": ["git"],
    "GitHub": ["github"],
    "GitLab": ["gitlab"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous deployment"],
    "Linux": ["linux"],
    "Selenium": ["selenium"],
    "PyTest": ["pytest"],
    "JUnit": ["junit"],
    "Automation Testing": ["automation testing", "test automation"],
    "Manual Testing": ["manual testing"],
    "Agile": ["agile"],
    "Scrum": ["scrum"],
    "Jira": ["jira"],
    "Project Management": ["project management"],
    "Stakeholder Management": ["stakeholder management"],
    "Leadership": ["leadership", "team lead", "people management"],
    "Communication": ["communication", "communication skills"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Science": ["data science"],
    "ETL": ["etl", "extract transform load"],
    "Spark": ["apache spark", "spark"],
    "Hadoop": ["hadoop"],
    "Kafka": ["kafka", "apache kafka"],
    "Airflow": ["airflow", "apache airflow"],
    "Snowflake": ["snowflake"],
    "Databricks": ["databricks"],
    "Salesforce": ["salesforce"],
    "SAP": ["sap"],
}

MANDATORY_MARKERS = [
    "must have", "mandatory", "required", "essential", "minimum requirement",
    "should have", "need to have", "must possess", "minimum qualifications"
]

PREFERRED_MARKERS = [
    "preferred", "nice to have", "good to have", "desirable", "bonus",
    "advantage", "plus", "preferred qualifications"
]

SECTION_HEADERS = {
    "summary": ["summary", "professional summary", "profile", "objective"],
    "experience": ["experience", "work experience", "employment history", "professional experience"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
    "education": ["education", "academic background", "qualifications"],
    "projects": ["projects", "key projects", "project experience"],
}


@dataclass
class MatchResult:
    job_title: str
    company: str
    overall_score: float
    semantic_score: float
    required_skill_score: float
    preferred_skill_score: float
    keyword_score: float
    experience_score: float
    ats_score: float
    decision: str
    required_skills: str
    matched_required_skills: str
    missing_required_skills: str
    preferred_skills: str
    missing_preferred_skills: str
    required_experience_years: str
    estimated_resume_experience_years: float
    top_missing_keywords: str
    recommendations: str


@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_exists(phrase: str, text: str) -> bool:
    phrase = normalize_text(phrase)
    text = normalize_text(text)
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_resume_text(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf_text(data)
    if name.endswith(".docx"):
        return extract_docx_text(data)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported resume type. Use PDF, DOCX, or TXT.")


def extract_skills(text: str) -> Set[str]:
    found = set()
    normalized = normalize_text(text)
    for canonical, aliases in SKILL_ALIASES.items():
        if any(phrase_exists(alias, normalized) for alias in aliases):
            found.add(canonical)
    return found


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def classify_job_skills(job_description: str) -> Tuple[Set[str], Set[str], Set[str]]:
    all_skills = extract_skills(job_description)
    required, preferred = set(), set()

    for sentence in split_sentences(job_description):
        sentence_skills = extract_skills(sentence)
        lower = sentence.lower()

        if any(marker in lower for marker in PREFERRED_MARKERS):
            preferred.update(sentence_skills)
        elif any(marker in lower for marker in MANDATORY_MARKERS):
            required.update(sentence_skills)

    unclassified = all_skills - required - preferred

    # Conservative default: skills not explicitly marked preferred are treated as required.
    required.update(unclassified)
    preferred -= required

    return required, preferred, all_skills


def extract_years_ranges(text: str) -> List[Tuple[float, float]]:
    t = normalize_text(text)
    ranges = []

    for lo, hi in re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:years|yrs)", t):
        ranges.append((float(lo), float(hi)))

    for num in re.findall(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years|yrs)", t):
        val = float(num)
        ranges.append((val, val))

    for num in re.findall(r"(?:minimum|min\.?|at least)\s*(\d+(?:\.\d+)?)\s*(?:years|yrs)", t):
        val = float(num)
        ranges.append((val, val))

    return ranges


def required_experience(job_description: str) -> Tuple[float, float]:
    ranges = extract_years_ranges(job_description)
    if not ranges:
        return 0.0, 0.0
    # Use the highest minimum requirement found.
    return max(ranges, key=lambda x: x[0])


def estimate_resume_experience(resume_text: str) -> float:
    t = normalize_text(resume_text)

    explicit = []
    for num in re.findall(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years|yrs)\s+(?:of\s+)?experience", t):
        explicit.append(float(num))
    for num in re.findall(r"(\d+(?:\.\d+)?)\s*(?:years|yrs)\s+(?:of\s+)?experience", t):
        explicit.append(float(num))
    if explicit:
        return max(explicit)

    # Approximate from year ranges when explicit experience is absent.
    year_pairs = re.findall(r"\b(19\d{2}|20\d{2})\s*(?:-|to)\s*(19\d{2}|20\d{2}|present|current)\b", t)
    total = 0
    current_year = pd.Timestamp.now().year
    for start, end in year_pairs:
        end_year = current_year if end in {"present", "current"} else int(end)
        duration = max(0, end_year - int(start))
        total += duration

    return float(min(total, 40))


def semantic_score(resume_text: str, jd_text: str, model: SentenceTransformer) -> float:
    embeddings = model.encode([resume_text, jd_text], normalize_embeddings=True)
    score = float(np.dot(embeddings[0], embeddings[1]))
    return round(max(0.0, min(1.0, score)) * 100, 2)


def coverage_score(resume_skills: Set[str], job_skills: Set[str], empty_value: float) -> float:
    if not job_skills:
        return empty_value
    return round(len(resume_skills & job_skills) / len(job_skills) * 100, 2)


def extract_keywords(text: str, limit: int = 40) -> List[str]:
    words = re.findall(r"\b[a-z][a-z0-9+#.-]{2,}\b", normalize_text(text))
    ignore = set(ENGLISH_STOP_WORDS).union({
        "job", "role", "candidate", "company", "team", "work", "working",
        "experience", "years", "skills", "skill", "required", "preferred",
        "responsibilities", "responsibility", "knowledge", "strong", "good",
        "ability", "using", "including", "position", "description", "looking"
    })
    freq: Dict[str, int] = {}
    for word in words:
        if word not in ignore and not word.isdigit():
            freq[word] = freq.get(word, 0) + 1
    return sorted(freq, key=lambda w: (freq[w], len(w)), reverse=True)[:limit]


def keyword_coverage(resume_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
    keywords = extract_keywords(jd_text)
    normalized_resume = normalize_text(resume_text)
    matched = [k for k in keywords if phrase_exists(k, normalized_resume)]
    missing = [k for k in keywords if k not in matched]
    score = len(matched) / len(keywords) * 100 if keywords else 50.0
    return round(score, 2), matched, missing


def ats_score(resume_text: str) -> Tuple[float, List[str]]:
    text = resume_text.lower()
    score = 100.0
    issues = []

    if len(resume_text.strip()) < 800:
        score -= 15
        issues.append("Resume appears too short for reliable ATS parsing.")

    detected_sections = 0
    for aliases in SECTION_HEADERS.values():
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            detected_sections += 1

    if detected_sections < 3:
        score -= 20
        issues.append("Use clear sections such as Summary, Experience, Skills, Education, and Projects.")

    if "@" not in resume_text:
        score -= 10
        issues.append("Email address was not detected.")

    if not re.search(r"(?:\+?\d[\d\s\-()]{8,}\d)", resume_text):
        score -= 10
        issues.append("Phone number was not detected.")

    bullet_count = len(re.findall(r"(^|\n)\s*(?:[-•*]|\d+\.)\s+", resume_text))
    if bullet_count < 3:
        score -= 10
        issues.append("Use concise bullet points for work achievements.")

    quantified = len(re.findall(r"\b\d+(?:\.\d+)?%|\b\d+\s*(?:users|clients|projects|members|days|hours|months|years)\b", text))
    if quantified < 2:
        score -= 10
        issues.append("Add measurable achievements where truthful.")

    if len(resume_text) > 12000:
        score -= 10
        issues.append("Resume may be too long; keep content focused.")

    return round(max(0.0, score), 2), issues


def experience_match_score(required_min: float, estimated: float) -> float:
    if required_min <= 0:
        return 70.0
    if estimated >= required_min:
        return 100.0
    if estimated <= 0:
        return 20.0
    return round(max(20.0, estimated / required_min * 100), 2)


def build_recommendations(
    missing_required: Set[str],
    missing_preferred: Set[str],
    missing_keywords: List[str],
    ats_issues: List[str],
    required_min: float,
    estimated_exp: float,
) -> List[str]:
    recs = []

    if missing_required:
        recs.append(
            "Verify whether the candidate genuinely has these required skills and, if so, add evidence in relevant projects or experience: "
            + ", ".join(sorted(missing_required))
        )

    if missing_preferred:
        recs.append(
            "Optional skills that may strengthen the application: "
            + ", ".join(sorted(missing_preferred))
        )

    if required_min and estimated_exp < required_min:
        recs.append(
            f"The job appears to require at least {required_min:g} years, while the resume shows approximately {estimated_exp:g}. Review before applying."
        )

    if missing_keywords:
        recs.append(
            "Consider naturally including truthful job-language terms such as: "
            + ", ".join(missing_keywords[:10])
        )

    recs.extend(ats_issues[:3])

    if not recs:
        recs.append("The resume already aligns well. Review the final application manually before submitting.")

    return recs


def evaluate_job(
    resume_text: str,
    jd_text: str,
    model: SentenceTransformer,
    threshold: float,
    job_title: str = "",
    company: str = "",
) -> MatchResult:
    resume_skills = extract_skills(resume_text)
    required, preferred, _ = classify_job_skills(jd_text)

    sem = semantic_score(resume_text, jd_text, model)
    required_score = coverage_score(resume_skills, required, 60.0)
    preferred_score = coverage_score(resume_skills, preferred, 70.0)
    keyword_score, _, missing_keywords = keyword_coverage(resume_text, jd_text)

    req_min, req_max = required_experience(jd_text)
    estimated_exp = estimate_resume_experience(resume_text)
    exp_score = experience_match_score(req_min, estimated_exp)

    ats, ats_issues = ats_score(resume_text)

    # Required skills dominate the decision.
    overall = (
        required_score * 0.35
        + sem * 0.25
        + keyword_score * 0.15
        + exp_score * 0.15
        + ats * 0.07
        + preferred_score * 0.03
    )
    overall = round(overall, 2)

    matched_required = resume_skills & required
    missing_required = required - resume_skills
    missing_preferred = preferred - resume_skills

    hard_block = bool(missing_required) and required_score < 60
    experience_block = req_min > 0 and estimated_exp > 0 and estimated_exp < req_min * 0.7

    if overall >= threshold and not hard_block and not experience_block:
        decision = "APPLY"
    elif overall >= threshold - 10:
        decision = "REVIEW"
    else:
        decision = "SKIP"

    recs = build_recommendations(
        missing_required,
        missing_preferred,
        missing_keywords,
        ats_issues,
        req_min,
        estimated_exp,
    )

    req_exp_text = "Not detected" if req_min == 0 else (
        f"{req_min:g}+ years" if req_min == req_max else f"{req_min:g}-{req_max:g} years"
    )

    return MatchResult(
        job_title=job_title or "Untitled role",
        company=company or "",
        overall_score=overall,
        semantic_score=sem,
        required_skill_score=required_score,
        preferred_skill_score=preferred_score,
        keyword_score=keyword_score,
        experience_score=exp_score,
        ats_score=ats,
        decision=decision,
        required_skills=", ".join(sorted(required)) or "Not detected",
        matched_required_skills=", ".join(sorted(matched_required)) or "None",
        missing_required_skills=", ".join(sorted(missing_required)) or "None",
        preferred_skills=", ".join(sorted(preferred)) or "Not detected",
        missing_preferred_skills=", ".join(sorted(missing_preferred)) or "None",
        required_experience_years=req_exp_text,
        estimated_resume_experience_years=estimated_exp,
        top_missing_keywords=", ".join(missing_keywords[:15]) or "None",
        recommendations=" | ".join(recs),
    )


def make_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Job Matches")
        ws = writer.book["Job Matches"]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 45)
    return output.getvalue()


def show_result(result: MatchResult) -> None:
    st.subheader(f"{result.job_title}" + (f" — {result.company}" if result.company else ""))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall", f"{result.overall_score}%")
    c2.metric("Required skills", f"{result.required_skill_score}%")
    c3.metric("Experience", f"{result.experience_score}%")
    c4.metric("ATS quality", f"{result.ats_score}%")

    st.progress(min(result.overall_score, 100) / 100)

    if result.decision == "APPLY":
        st.success("Recommendation: APPLY")
    elif result.decision == "REVIEW":
        st.warning("Recommendation: REVIEW MANUALLY")
    else:
        st.error("Recommendation: SKIP")

    left, right = st.columns(2)
    with left:
        st.markdown("**Matched required skills**")
        st.write(result.matched_required_skills)
        st.markdown("**Missing required skills**")
        st.write(result.missing_required_skills)
        st.markdown("**Required experience**")
        st.write(result.required_experience_years)

    with right:
        st.markdown("**Preferred skills**")
        st.write(result.preferred_skills)
        st.markdown("**Missing preferred skills**")
        st.write(result.missing_preferred_skills)
        st.markdown("**Estimated resume experience**")
        st.write(f"{result.estimated_resume_experience_years:g} years")

    st.markdown("**Top missing keywords**")
    st.write(result.top_missing_keywords)

    st.markdown("**Resume tailoring suggestions**")
    for recommendation in result.recommendations.split(" | "):
        st.write(f"- {recommendation}")

    st.caption(
        "Do not add skills, qualifications, employment, or results that the candidate does not actually have."
    )


st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
st.title(APP_TITLE)
st.caption(
    "Compare a resume against one job description or a batch of jobs. "
    "The score is a decision-support heuristic, not an official Naukri or recruiter score."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Apply threshold", 50, 95, 80)
    st.markdown(
        """
        **Score weights**
        - Required skills: 35%
        - Semantic match: 25%
        - Keywords: 15%
        - Experience: 15%
        - ATS quality: 7%
        - Preferred skills: 3%
        """
    )
    st.info("Paste job descriptions or upload a CSV. This version does not scrape or auto-apply on Naukri.")

resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])

if resume_file:
    try:
        resume_text = extract_resume_text(resume_file)
        if len(normalize_text(resume_text)) < 100:
            st.error("Very little text was extracted. Use a text-based PDF, DOCX, or TXT resume.")
            st.stop()

        model = load_model()

        tab1, tab2, tab3 = st.tabs(["Single Job", "Batch Jobs", "Resume Audit"])

        with tab1:
            job_title = st.text_input("Job title")
            company = st.text_input("Company")
            jd = st.text_area("Paste the complete job description", height=340)

            if st.button("Analyze job", type="primary", use_container_width=True):
                if not jd.strip():
                    st.error("Paste a job description first.")
                else:
                    result = evaluate_job(
                        resume_text, jd, model, threshold, job_title, company
                    )
                    show_result(result)

                    result_df = pd.DataFrame([asdict(result)])
                    st.download_button(
                        "Download result as CSV",
                        result_df.to_csv(index=False).encode("utf-8"),
                        file_name="job_match_result.csv",
                        mime="text/csv",
                    )

        with tab2:
            st.write(
                "Upload a CSV with columns: `job_title`, `company`, and `job_description`."
            )
            batch_file = st.file_uploader("Upload jobs CSV", type=["csv"], key="batch")

            if batch_file is not None:
                jobs_df = pd.read_csv(batch_file)
                required_columns = {"job_title", "company", "job_description"}

                if not required_columns.issubset(jobs_df.columns):
                    st.error("CSV must contain job_title, company, and job_description columns.")
                elif st.button("Analyze all jobs", type="primary", use_container_width=True):
                    results = []
                    progress = st.progress(0)

                    for index, row in jobs_df.iterrows():
                        result = evaluate_job(
                            resume_text,
                            str(row["job_description"]),
                            model,
                            threshold,
                            str(row["job_title"]),
                            str(row["company"]),
                        )
                        results.append(asdict(result))
                        progress.progress((index + 1) / len(jobs_df))

                    result_df = pd.DataFrame(results).sort_values(
                        ["overall_score", "required_skill_score"],
                        ascending=False,
                    )

                    st.dataframe(
                        result_df[
                            [
                                "job_title",
                                "company",
                                "overall_score",
                                "decision",
                                "required_skill_score",
                                "experience_score",
                                "ats_score",
                                "missing_required_skills",
                            ]
                        ],
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download batch results as CSV",
                        result_df.to_csv(index=False).encode("utf-8"),
                        file_name="batch_job_match_results.csv",
                        mime="text/csv",
                    )

                    st.download_button(
                        "Download batch results as Excel",
                        make_excel_bytes(result_df),
                        file_name="batch_job_match_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        with tab3:
            score, issues = ats_score(resume_text)
            resume_skills = sorted(extract_skills(resume_text))
            estimated_exp = estimate_resume_experience(resume_text)

            c1, c2, c3 = st.columns(3)
            c1.metric("ATS quality", f"{score}%")
            c2.metric("Detected skills", len(resume_skills))
            c3.metric("Estimated experience", f"{estimated_exp:g} years")

            st.markdown("**Detected skills**")
            st.write(", ".join(resume_skills) or "No skills detected from the current dictionary.")

            st.markdown("**ATS improvement suggestions**")
            if issues:
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                st.success("No major ATS-formatting issues were detected.")

            with st.expander("Preview extracted resume text"):
                st.text(resume_text[:15000])

    except Exception as exc:
        st.error(f"Unable to process the resume: {exc}")
else:
    st.info("Upload a resume to begin.")
