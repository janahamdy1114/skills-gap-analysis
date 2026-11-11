
from pathlib import Path
import pandas as pd


def parse_job_file(path):
    """Read one Markdown job description and split it into sections."""
    text = Path(path).read_text(encoding="utf-8")

    job = {"source_file": Path(path).name, "sections": {}}
    current_section = None

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue
        elif line.startswith("Title:"):
            job["job_title"] = line.replace("Title:", "").strip()
        elif line.startswith("Company:"):
            job["company"] = line.replace("Company:", "").strip()
        elif line.startswith("Location:"):
            job["location"] = line.replace("Location:", "").strip()
        elif line.startswith("## "):
            current_section = line.replace("## ", "").strip().lower()
            job["sections"][current_section] = []
        elif current_section:
            job["sections"][current_section].append(line.lstrip("- ").strip())

    return job


def load_all_jobs(folder="data/synthetic"):
    """Load synthetic Markdown job descriptions."""
    jobs = []

    for path in sorted(Path(folder).glob("*.md")):
        jobs.append(parse_job_file(path))

    return jobs


def load_csv_jobs(path="data/processed/selected_30_jobs.csv"):
    """Load real job descriptions from the selected CSV dataset."""
    df = pd.read_csv(path)

    jobs = []

    for index, row in df.iterrows():
        jobs.append({
            "job_id": f"job_{index + 1:03d}",
            "job_title": row["job_title"],
            "company": row["company_name"],
            "description": row["cleaned_description"],
            "role_category": row["role_category"],
            "seniority_level": row["seniority_level"],
            "location": (
                f"{row['company_address_locality']}, "
                f"{row['company_address_region']}"
            ),
            "source_file": "selected_30_jobs.csv"
        })

    return jobs


if __name__ == "__main__":
    jobs = load_csv_jobs()

    print(f"Loaded {len(jobs)} real job descriptions:\n")

    for job in jobs:
        print(
            f"- {job['job_id']}: "
            f"{job['job_title']} at {job['company']}"
        )
