# Professional Resume–Job Matcher

A local Streamlit application for comparing a resume with one or many job descriptions.

## Features

- PDF, DOCX, and TXT resume parsing
- Semantic similarity using Sentence Transformers
- Required vs preferred skill detection
- Experience requirement comparison
- ATS-format quality checks
- Missing skill and keyword analysis
- Apply / Review / Skip recommendation
- Batch job comparison from CSV
- CSV and Excel export
- Resume-tailoring suggestions without inventing experience

## Important limitation

This tool does not scrape Naukri.com and does not automatically apply for jobs. Paste job descriptions manually or upload a CSV. Automated scraping or applying may violate platform terms and may expose account credentials.

## Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The model downloads on the first run.

## Batch CSV format

Use these exact columns:

```csv
job_title,company,job_description
Python Developer,Example Ltd,"We require Python, SQL, AWS and 3+ years of experience."
Data Analyst,Demo Corp,"Must have SQL, Excel and Power BI. Python is preferred."
```

A sample file is included.

## Scoring

- Required skills: 35%
- Semantic similarity: 25%
- Keyword coverage: 15%
- Experience match: 15%
- ATS quality: 7%
- Preferred skills: 3%

The default Apply threshold is 80%.

## Accuracy guidance

Treat the score as decision support only. Always manually verify:

- Mandatory qualifications
- Location and work authorization
- Notice period
- Salary expectations
- Certifications
- Shift requirements
- Employment gaps
- Role seniority

Never add untrue skills or experience to a resume.
