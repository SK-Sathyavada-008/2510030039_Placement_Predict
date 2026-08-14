import os
import pandas as pd


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    Loads the CSV file and validates that all required columns exist.
    """

    # Check whether file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Critical Error: Targeted data footprint not discovered at {file_path}"
        )

    print(f"Executing secure data extraction from: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Support the dataset's actual column names while preserving the
    # standardized names expected by the ingestion pipeline.
    column_aliases = {
        "coding_skills": ["coding_skills", "coding_skill_score", "coding_skill"],
        "communication_skills": [
            "communication_skills",
            "communication_skill_score",
            "communication_skill",
        ],
        "internships": ["internships", "internships_count", "internship_count"],
    }

    for standardized_name, aliases in column_aliases.items():
        matching_col = next((col for col in aliases if col in df.columns), None)
        if matching_col and matching_col != standardized_name:
            df.rename(columns={matching_col: standardized_name}, inplace=True)

    # Required columns
    required_columns = [
        "branch",
        "college_tier",
        "cgpa",
        "backlogs",
        "coding_skills",
        "communication_skills",
        "internships",
        "projects_count",
        "placement_status",
        "salary_package_lpa"
    ]

    # Find missing columns
    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    # Raise error if any required columns are missing
    if missing_cols:
        raise ValueError(
            f"Schema Validation Failure: Missing essential feature targets: {missing_cols}"
        )

    print(
        f"Data ingestion resolved successfully. "
        f"Dimensions captured: {df.shape[0]} samples, {df.shape[1]} metrics."
    )

    return df


if __name__ == "__main__":

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    try:
        raw_data = load_and_validate_data(DATA_PATH)

    except Exception as e:
        print(f"Ingestion lifecycle termination: {str(e)}")