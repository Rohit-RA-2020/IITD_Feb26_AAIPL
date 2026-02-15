# Meta-Llama-3.1-8B via Unsloth FastLanguageModel.
import time
import torch
from pathlib import Path
from typing import Optional, Union, List
from unsloth import FastLanguageModel

torch.random.manual_seed(0)

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

class QAgent(object):
    def __init__(self, **kwargs):
        self.max_seq_length = kwargs.get('max_seq_length', 4096)
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

        model_name = str(Path(__file__).parent.parent / "hf_models" / "Meta-Llama-3.1-8B-Instruct")

        # load the tokenizer and the model via Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map="cuda",
        )
        # Enable native 2x faster inference
        FastLanguageModel.for_inference(self.model)

    def generate_response(self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]

        # Format all messages using the alpaca prompt template
        texts = []
        for msg in message:
            text = ALPACA_PROMPT.format(
                system_prompt,  # instruction
                msg,            # input
                "",             # output - leave blank for generation
            )
            texts.append(text)

        # Tokenize all texts together with padding
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show_var = kwargs.get('tgps_show', False)
        max_new_tokens = kwargs.get('max_new_tokens', 1024)

        # Conduct batch text completion
        if tgps_show_var:
            start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if tgps_show_var:
            generation_time = time.time() - start_time

        # Decode the batch
        batch_outs = []
        if tgps_show_var:
            token_len = 0
        for input_ids, generated_sequence in zip(model_inputs.input_ids, generated_ids):
            # Extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids):]

            # Compute total tokens generated
            if tgps_show_var:
                token_len += len(output_ids)

            # Decode the generated output
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            batch_outs.append(content)

        if tgps_show_var:
            return batch_outs[0] if len(batch_outs) == 1 else batch_outs, token_len, generation_time
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None

if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """
    
    response, tl, tm = model.generate_response(prompt, tgps_show=True, max_new_tokens=512)
    print("Single example response:")
    print("Response: ", response)
    print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(prompts, tgps_show=True, max_new_tokens=512)
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    print(f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec")
