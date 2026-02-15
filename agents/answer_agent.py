#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent


class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""

    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1

    def build_prompt(self, question_data: Dict[str, str | Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""

        sys_prompt1 = "You are an expert in quantitative aptitude for competitive exams, solving MCQs with step-by-step reasoning before selecting the correct answer."
        sys_prompt2 = (
            "You are an expert answer agent specializing in solving multiple-choice questions (MCQs) that test "
            "quantitative aptitude skills, as seen in top-tier competitive exams. "
            "You have a deep understanding of logical reasoning, puzzles, and analytical problem-solving under exam conditions. "
            "For each question, think step by step using a clear chain-of-thought approach. "
            "Break down the problem, analyze all options, eliminate distractors, and then confidently select the correct answer. "
            "Always explain your reasoning before finalizing your choice."
        )

        # IMPROVED: Match the training format exactly
        tmpl = (
            "Topic: {}\n\n"
            "Question: {}\n\n"
            "Choices:\n{}\n"
        )

        prompt = tmpl.format(
            question_data.get("topic", "General"),
            question_data["question"],
            self._format_choices_for_training(question_data["choices"])
        )

        return prompt, sys_prompt1 if self.select_prompt1 else sys_prompt2

    def _parse_model_output(self, output: str) -> Dict[str, str]:
        """Parse model output to extract JSON, handling various formats"""
        
        # If already a dict, return it
        if isinstance(output, dict):
            return output
        
        # If it's a string, try to parse JSON
        if isinstance(output, str):
            # Remove any markdown code blocks
            output = re.sub(r'```json\s*', '', output)
            output = re.sub(r'```\s*', '', output)
            
            # Try to find JSON object
            if '{' in output and '}' in output:
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                json_str = output[json_start:json_end]
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # If no JSON found, try to extract answer letter
            answer_match = re.search(r'"answer"\s*:\s*"([A-D])"', output)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', output)
            
            if answer_match:
                return {
                    "answer": answer_match.group(1),
                    "reasoning": reasoning_match.group(1) if reasoning_match else ""
                }
        
        # Fallback: return error
        return {"answer": "ERROR", "reasoning": f"Could not parse: {output}"}

    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[Dict], int | None, float | None]:
        """Generate answer(s) for the given question(s)"""
        
        is_batch = isinstance(question_data, list)
        
        if is_batch:
            prompts = []
            sys_prompts = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompts.append(p)
                sys_prompts.append(sp)
            
            # Use first system prompt for all (they're the same)
            resp, tl, gt = self.agent.generate_response(prompts, sys_prompts[0], **kwargs)
        else:
            prompt, sp = self.build_prompt(question_data)
            resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)
        
        # Parse responses
        if is_batch:
            if isinstance(resp, list):
                parsed_responses = [self._parse_model_output(r) for r in resp]
            else:
                # Single response for batch (shouldn't happen)
                parsed_responses = [self._parse_model_output(resp)]
        else:
            parsed_responses = self._parse_model_output(resp)
        
        return parsed_responses, tl, gt

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            
            # Ensure batch_answers is a list
            if not isinstance(batch_answers, list):
                batch_answers = [batch_answers]
            
            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        pbar.close()
        return answers, tls, gts

    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(self, ans: List[str | Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""

        def basic_checks(a1: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["answer"]
            if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                # Check if answer is valid
                if a1["answer"] not in "ABCDabcd":
                    return False
                if len(a1["answer"]) != 1:
                    return False
                    
                check_len = self.count_tokens_a(a1["answer"])
                if check_len < 50:
                    check_len += self.count_tokens_a(a1.get("reasoning", "None"))
                    if check_len < 512:
                        return True
            return False

        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Try to parse JSON string
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at index {i}: {a}")
                    filtered_answers.append(None)
            else:
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers

    def save_answers(self, answers: List[str], file_path: str | Path) -> None:
        """Save generated answers to a JSON file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([a for a in answers], f, indent=4)

    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            if not re.match(r"^[A-D]\)", choice.strip()):
                letter = chr(65 + len(formatted))
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)
    
    def _format_choices_for_training(self, choices: List[str]) -> str:
        """Format choices exactly like training data"""
        formatted = []
        for i, choice in enumerate(choices):
            # Ensure consistent format: "A) text"
            if not choice.strip().startswith(chr(65 + i) + ")"):
                formatted.append(f"{chr(65 + i)}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return "\n".join(formatted)


# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse

    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument(
        "--input_file",
        type=str,
        default="outputs/filtered_questions.json",
        help="Path to the input JSON file with questions",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/answers.json",
        help="Path to save the answers",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for processing questions"
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = argparser.parse_args()

    SELECT_PROMPT1 = False

    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1)

    gen_kwargs = {"tgps_show": True}
    try:
        with open("agen.yaml", "r") as f:
            gen_kwargs.update(yaml.safe_load(f))
    except FileNotFoundError:
        print("⚠️  agen.yaml not found, using defaults")
        gen_kwargs.update({
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": True
        })
    
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions, batch_size=args.batch_size, **gen_kwargs
    )

    # Answers are already parsed as dicts, no need for additional parsing
    if args.verbose:
        for idx, (q, a) in enumerate(zip(sample_questions, answer)):
            print(f"\n=== Question {idx+1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Model Answer: {a}")
            if isinstance(a, dict):
                print(f"  Answer: {a.get('answer', 'N/A')}")
                print(f"  Reasoning: {a.get('reasoning', 'N/A')}")

    if args.verbose and gen_kwargs.get("tgps_show", False):
        for idx, (tl, gt) in enumerate(zip(tls, gts)):
            if tl and gt:
                print(f"BATCH - {idx}: Tokens: {tl}, Time: {gt:.3f}s, TGPS: {tl/gt:.3f}")

    # Save answers
    agent.save_answers(answer, args.output_file)
    filtered_file_name = args.output_file.replace("answers.json", "filtered_answers.json")
    agent.save_answers(agent.filter_answers(answer), filtered_file_name)
    
    print(f"\n✅ Saved {len(answer)} answers to {args.output_file}")
    print(f"✅ Saved {len([a for a in answer if a])} filtered answers to {filtered_file_name}")