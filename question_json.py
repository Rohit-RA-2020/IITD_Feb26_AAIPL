import json
from pathlib import Path

# Paths
# input_path = "./output/filtered_questions.json"
input_path = "/workspace/AAIPL/outputs/filtered_questions.json"
# output_path = "./data/final/question_agent_train.json"
output_path = "/workspace/AAIPL/data/final/question_agent_train.json"

# Load raw MCQ dataset
with open(input_path, "r") as f:
    raw_data = json.load(f)

converted_data = []

for item in raw_data:
    topic = item["topic"]

    # Build assistant JSON (what Q-Agent should output)
    assistant_json = {
        "topic": item["topic"],
        "question": item["question"],
        "choices": item["choices"],
        "answer": item["answer"],
        "explanation": item["explanation"]
    }

    # Create user instruction prompt
    user_prompt = f"""
Generate a high-quality multiple choice question on the topic: {topic}.

Return your response strictly as a valid JSON object with this structure:

{{
    "topic": "{topic}",
    "question": "Question text ending with a question mark?",
    "choices": [
        "A) Option",
        "B) Option",
        "C) Option",
        "D) Option"
    ],
    "answer": "A",
    "explanation": "Clear reasoning explanation"
}}
"""

    converted_data.append({
        "conversations": [
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}
        ]
    })

# Create output folder
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# Save final Q-Agent training dataset
with open(output_path, "w") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print("✅ question_agent_train.json created successfully!")
print(f"Total training examples: {len(converted_data)}")
