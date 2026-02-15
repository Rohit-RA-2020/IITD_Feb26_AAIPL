import json
from pathlib import Path

DATASETS_DIR = Path("/workspace/AAIPL/DATASETS")
OUTPUT_DIR = Path("data/question_agent")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔄 Converting existing datasets to Q-Agent format...")

# Load all questions
all_questions = []
for json_file in DATASETS_DIR.glob("*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    all_questions.extend(data)

print(f"✅ Loaded {len(all_questions)} questions")

# Convert to training format
training_data = []
for q in all_questions:
    conversation = {
        "conversations": [
            {
                "role": "user",
                "content": f"Generate a challenging {q['topic']} question in JSON format."
            },
            {
                "role": "assistant",
                "content": json.dumps(q, ensure_ascii=False)
            }
        ]
    }
    training_data.append(conversation)

# Save
training_file = OUTPUT_DIR / "question_agent_train.json"
with open(training_file, 'w') as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(training_data)} training examples to {training_file}")
print(f"🚀 Ready to train Q-Agent!")