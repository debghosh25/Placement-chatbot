import os
import pickle
from pathlib import Path

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SQL_DUMP_CANDIDATES = [
    PROJECT_DIR / "placement_db_export.sql",
    PROJECT_DIR / "placement_db.sql",
    BASE_DIR / "placement_db_export.sql",
    BASE_DIR / "placement_db.sql",
]


def find_sql_dump() -> Path:
    for path in SQL_DUMP_CANDIDATES:
        if path.exists():
            return path

    matches = sorted(PROJECT_DIR.glob("placement_db*.sql"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "No placement database SQL dump found. Put placement_db_export.sql "
        "in the Chatbot_V3-main folder."
    )


def parse_copy_table(sql_path: Path, table_name: str) -> list[dict]:
    rows = []
    copy_prefix = f"COPY public.{table_name} "
    in_copy = False
    columns = []

    with sql_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not in_copy:
                if line.startswith(copy_prefix):
                    column_block = line.split("(", 1)[1].split(")", 1)[0]
                    columns = [col.strip() for col in column_block.split(",")]
                    in_copy = True
                continue

            if line == r"\.":
                break

            values = [None if value == r"\N" else value for value in line.split("\t")]
            rows.append(dict(zip(columns, values)))

    return rows


def placement_to_doc(row: dict) -> dict:
    year = str(row.get("academic_year") or "").strip()
    stream = str(row.get("stream") or "").strip()
    company = str(row.get("company") or "").strip()

    text_parts = [
        ("Roll", row.get("roll")),
        ("Student Name", row.get("student_name")),
        ("Stream", stream),
        ("Company", company),
        ("Academic Year", year),
        ("Source File", row.get("source_file")),
    ]
    text = " | ".join(
        f"{label}: {value}"
        for label, value in text_parts
        if value not in (None, "")
    )

    return {
        "text": text,
        "year": year,
        "department": stream,
        "company": company,
        "doc_type": "placement",
    }


def package_to_doc(row: dict) -> dict:
    year = str(row.get("academic_year") or "").strip()
    company = str(row.get("company") or "").strip()
    package_amount = str(row.get("package_amount") or "").strip()

    text_parts = [
        ("Company", company),
        ("Academic Year", year),
        ("Package Amount", package_amount),
        ("Source File", row.get("source_file")),
    ]
    text = " | ".join(
        f"{label}: {value}"
        for label, value in text_parts
        if value not in (None, "")
    )

    return {
        "text": text,
        "year": year,
        "department": "",
        "company": company,
        "package_amount": package_amount,
        "doc_type": "placement_package",
    }


def main():
    sql_path = find_sql_dump()
    print(f"Loading placement records from {sql_path}")

    placement_rows = parse_copy_table(sql_path, "placements")
    package_rows = parse_copy_table(sql_path, "placement_packages")

    all_docs = (
        [placement_to_doc(row) for row in placement_rows]
        + [package_to_doc(row) for row in package_rows]
    )
    all_docs = [doc for doc in all_docs if doc["text"].strip()]

    if not all_docs:
        raise ValueError("No records found in SQL dump.")

    print(f"Loaded {len(placement_rows)} placement records")
    print(f"Loaded {len(package_rows)} package records")
    print(f"Building index from {len(all_docs)} SQL records")

    vectorizer = TfidfVectorizer()
    texts = [doc["text"] for doc in all_docs]
    vectors = vectorizer.fit_transform(texts)
    embeddings = vectors.toarray().astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("FAISS index created")

    index_dir = BASE_DIR / "index"
    os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, str(index_dir / "faiss.index"))

    with (index_dir / "docs.pkl").open("wb") as f:
        pickle.dump(all_docs, f)

    with (index_dir / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    print("Index saved successfully!")


if __name__ == "__main__":
    main()
