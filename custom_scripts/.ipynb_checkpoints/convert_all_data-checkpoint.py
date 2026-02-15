import json
from pathlib import Path

# Your directory path
DATA_DIR = Path("/workspace/AAIPL/DATASETS")

# Output directory
OUTPUT_DIR = Path("data/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all JSON files
json_files = list(DATA_DIR.glob("*.json"))
print(f"📁 Found {len(json_files)} JSON files in {DATA_DIR}")

# Store all converted data
all_chatML_data = []

# Process each file
for json_file in json_files:
    print(f"Processing: {json_file.name}")
    
    try:
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle both single dict and list of dicts
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        
        # Convert each item
        for item in raw_data:
            # Format the question with choices
            user_message = f"""Topic: {item['topic']}

Question: {item['question']}

Choices:
{item['choices'][0]}
{item['choices'][1]}
{item['choices'][2]}
{item['choices'][3]}"""
            
            # Format the answer in competition format
            assistant_message = json.dumps({
                "answer": item['answer'],
                "reasoning": item['explanation']
            })
            
            # Create conversation structure
            conversation = {
                "conversations": [
                    {
                        "role": "user",
                        "content": user_message
                    },
                    {
                        "role": "assistant",
                        "content": assistant_message
                    }
                ]
            }
            
            all_chatML_data.append(conversation)
        
        print(f"  ✅ Converted {len(raw_data)} examples from {json_file.name}")
    
    except Exception as e:
        print(f"  ❌ Error processing {json_file.name}: {e}")

# Save all data into one training file
output_file = OUTPUT_DIR / "answer_agent_train.json"
with open(output_file, 'w') as f:
    json.dump(all_chatML_data, f, indent=2)

print(f"\n🎉 Total converted: {len(all_chatML_data)} examples")
print(f"📁 Saved to: {output_file}")
