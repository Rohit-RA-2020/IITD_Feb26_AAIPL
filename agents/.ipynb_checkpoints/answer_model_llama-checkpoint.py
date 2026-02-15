# Meta-Llama-3.1-8B via Unsloth FastLanguageModel.
import time
import torch
from pathlib import Path
from typing import Optional, List
from unsloth import FastLanguageModel

torch.random.manual_seed(0)

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

class AAgent(object):
    def __init__(self, **kwargs):
        self.max_seq_length = kwargs.get('max_seq_length', 2048)
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
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response("Solve: 2x + 5 = 15", system_prompt="You are a math tutor.", tgps_show=True, max_new_tokens=512)
    print(f"Single response: {response}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")

    # Batch processing
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(messages, max_new_tokens=512, tgps_show=True)
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    print(f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}")
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story",
        max_new_tokens=512
    )
    print(f"Custom response: {response}")
