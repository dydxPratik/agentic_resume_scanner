import json
import os
from typing import Literal
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "openai/gpt-oss-20b"


# 1. Define both Pydantic Schemas
class Resume(BaseModel):
    candidate_name: str
    email: str | None = None
    phone: str | None = None
    education: list[str]
    experience: list[str]
    projects: list[str]
    technical_skills: list[str]
    soft_skills: list[str]
    certifications: list[str]


class Evaluation(BaseModel):
    candidate_name: str
    is_eligible: bool
    role_level: Literal[
        "Not Qualified", "Intern", "Junior", "Mid-Level", "Senior"
    ]
    match_score: int = Field(ge=0, le=100)
    missing_skills: list[str]
    summary: str


# 2. Read both input files
with open("requirements.txt", "r", encoding="utf-8") as file:
    hreq = file.read()

with open("resume.txt", "r", encoding="utf-8") as file:
    raw_resume = file.read()

# 3. Step 1: Extract Resume Info
extract_prompt = f"""Extract the essential information from the candidate resume below and respond strictly as valid JSON matching this schema.

Schema: {json.dumps(Resume.model_json_schema())}

Resume:
{raw_resume}
"""

resume_response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are an expert HR parser. Output only valid JSON.",
        },
        {"role": "user", "content": extract_prompt},
    ],
    response_format={"type": "json_object"},
)

candidate_resume = resume_response.choices[0].message.content

# 4. Step 2: Evaluate against HR Requirements
SYSTEM_PROMPT = """
You are a strict technical recruiter. Evaluate the candidate's resume strictly against the provided HR requirements.

Rules:
- Base your decision ONLY on explicit facts in the resume. Do not assume unstated skills.
- Assign the role level strictly based on their actual experience and matched skills:
  - "Not Qualified": Missing mandatory skills or wrong domain.
  - "Intern": Basic knowledge/education, no real projects or experience.
  - "Junior": Meets mandatory skills, 0–2 years experience (projects/internships count).
  - "Mid-Level": 2–5 years experience, strong mandatory + preferred skills.
  - "Senior": 5+ years experience, architecture/leadership.
- Output ONLY valid JSON matching the requested schema. No extra text.
"""

eval_prompt = f"""
Evaluate the extracted resume against the HR requirements and return valid JSON matching the schema below.

SCHEMA: {json.dumps(Evaluation.model_json_schema())}
RESUME: {candidate_resume}
HR REQUIREMENTS: {hreq}
"""

final_response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": eval_prompt},
    ],
    response_format={"type": "json_object"},
)

# Print just the text content (or validate with Evaluation.model_validate_json)
print(final_response.choices[0].message.content)