"""
Preprocessing pipeline for Shakespeare dataset.
"""

import json
from pathlib import Path


RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed/processed_records.json"


class ShakespearePreprocessor:
    def __init__(self):
        self.records = []

    def load_json_files(self):
        raw_path = Path(RAW_DATA_PATH)

        for file in raw_path.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    self.records.extend(data)

    def validate_records(self):
        required_fields = [
            "play",
            "act",
            "scene",
            "speaker",
            "text"
        ]

        valid_records = []

        for record in self.records:
            if all(field in record for field in required_fields):
                valid_records.append(record)

        self.records = valid_records

    def save_processed_records(self):
        output_path = Path(PROCESSED_DATA_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def run(self):
        self.load_json_files()
        self.validate_records()
        self.save_processed_records()

        print(f"Processed {len(self.records)} records")


if __name__ == "__main__":
    pipeline = ShakespearePreprocessor()
    pipeline.run()
