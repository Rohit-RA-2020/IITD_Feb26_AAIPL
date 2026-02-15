import json
from pathlib import Path

# ========= CONFIG =========
DATA_DIR = Path("/workspace/AAIPL/DATASETS")
CLEAN_DIR = Path("DATASETS/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_KEYS = {
    "topic": str,
    "question": str,
    "choices": list,
    "answer": str,
    "explanation": str
}
# ==========================


def is_valid_item(item):
    """
    Validate structure and data types.
    Returns (True, None) if valid.
    Returns (False, reason) if invalid.
    """

    # Must be dict
    if not isinstance(item, dict):
        return False, "Item is not a dictionary"

    # Check required keys
    for key, expected_type in REQUIRED_KEYS.items():
        if key not in item:
            return False, f"Missing key: {key}"

        if not isinstance(item[key], expected_type):
            return False, f"Wrong type for key: {key}"

        if not item[key]:  # Empty values
            return False, f"Empty value for key: {key}"

    # Validate choices length
    if len(item["choices"]) != 4:
        return False, "Choices must contain exactly 4 options"

    return True, None


# ========= PROCESS FILES =========

json_files = list(DATA_DIR.glob("*.json"))
print(f"\n📁 Found {len(json_files)} JSON files")

total_removed = 0
total_valid = 0

for json_file in json_files:
    print(f"\nProcessing: {json_file.name}")

    try:
        with open(json_file, "r") as f:
            raw_data = json.load(f)

        # Handle single object case
        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        cleaned_data = []
        removed_count = 0

        for idx, item in enumerate(raw_data):
            valid, reason = is_valid_item(item)

            if valid:
                cleaned_data.append(item)
            else:
                removed_count += 1
                print(f"  ❌ Removed item {idx}: {reason}")

        # Save cleaned file
        output_file = CLEAN_DIR / json_file.name
        with open(output_file, "w") as f:
            json.dump(cleaned_data, f, indent=2)

        print(f"  ✅ Valid items: {len(cleaned_data)}")
        print(f"  🗑 Removed items: {removed_count}")

        total_valid += len(cleaned_data)
        total_removed += removed_count

    except Exception as e:
        print(f"  🚨 Failed to process {json_file.name}: {e}")

print("\n============================")
print(f"🎉 Total valid items: {total_valid}")
print(f"🗑 Total removed items: {total_removed}")
print(f"📁 Clean files saved to: {CLEAN_DIR}")
print("============================\n")
