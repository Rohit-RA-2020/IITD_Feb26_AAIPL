#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent
# from .question_model_llama import QAgent

import random
import json
import re

class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    @staticmethod
    def _strip_choice_prefix(choice: str) -> str:
        if not isinstance(choice, str):
            return ""
        cleaned = choice.strip()
        # Remove one or more leading labels like "A) ", "B. ", etc.
        cleaned = re.sub(r"^(?:\s*[A-Da-d][\)\.\:\-]\s*)+", "", cleaned).strip()
        return cleaned

    @classmethod
    def _normalize_choices_with_labels(cls, choices: List[Any]) -> List[str] | None:
        if not isinstance(choices, list) or len(choices) != 4:
            return None
        labels = ["A", "B", "C", "D"]
        normalized = []
        for idx, choice in enumerate(choices):
            base = cls._strip_choice_prefix(str(choice))
            if not base:
                return None
            normalized.append(f"{labels[idx]}) {base}")
        return normalized

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        r"""
        Build a string of example questions from the provided samples.
        """
        if not inc_samples:
            return ""
        fmt = (
            "EXAMPLE: {}\n"
            "{{\n"
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["{}", "{}", "{}", "{}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            "}}"
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            raw_choices = sample.get("choices", [""] * 4)
            choices = self._normalize_choices_with_labels(raw_choices) or [
                "A) Option 1",
                "B) Option 2",
                "C) Option 3",
                "D) Option 4",
            ]
            answer = sample.get("answer", "")
            explanation = sample.get("explanation", "")
            sample_str += (
                fmt.format(
                    topic, topic.split("/")[-1], question, *choices, answer, explanation
                )
                + "\n\n"
            )
        return sample_str.strip()

    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""

        if wadvsys:
            # TODO: Manipulate this SYS prompt for better results
            sys_prompt = """
            You are an **expert-level examiner** with deep expertise in designing **highly challenging and conceptually rigorous multiple-choice questions (MCQs)** for the **Quantitative Aptitude and Analytical Reasoning** sections of top-tier competitive exams.
            Think step by step to generate the question and solve the same, but only output the final answer. Do not show your thinking process.
            **Please DO NOT reveal the solution steps or any intermediate reasoning.**
            """
        else:
            sys_prompt = "You are an examiner tasked with creating extremely difficult multiple-choice questions"
        tmpl = (
            "Generate an EXTREMELY DIFFICULT MCQ on topic: {0}.\n\n"
            "**CRITICAL REQUIREMENTS:**\n"
            '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
            "2.  **Question Quality**: The question must be EXTREMELY DIFFICULT, clear, and test deep conceptual understanding. Avoid trivial or ambiguous questions.\n"
            '3.  **Choices (4 total)**: Generate exactly FOUR multiple-choice options, labeled "A)", "B)", "C)", and "D)".\n'
            "4.  **Single Correct Answer**: Ensure that option {2} is only factually correct.\n"
            "5.  **Plausible Distractors**: While option {3} are three incorrect UNIQUE choices which are highly plausible and common misconceptions related to the topic, designed to mislead someone without expert knowledge.\n"
            '6.  **Answer Key**: The "answer" field in the JSON should be ONLY the letter {4}.\n'
            '7.  **Explanation**: The "explanation" field provides a concise (under 100 words) and clear justification for why the correct answer is correct.\n\n'
            "{5}"
            "RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n"
            "EXAMPLE: {6}\n"
            "{{\n"
            '  "topic": "{7}",\n'
            '  "question": "...",\n'
            '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            '  "answer": "{8}",\n'
            '  "explanation": "Provide a brief explanation why {9} is correct within 100 words."\n'
            "}}"
        )
        # Remove model's preferential bias for options
        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join(
            [opt for opt in ["A", "B", "C", "D"] if opt != correct_option]
        )

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""
        prompt = tmpl.format(
            topic,
            topic,
            correct_option,
            distractors,
            correct_option,
            inc_samples_ex,
            topic,
            topic.split("/")[-1],
            correct_option,
            correct_option,
        )

        return prompt, sys_prompt

    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        """Generate a question prompt for the LLM"""
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                p, sp = self.build_prompt(
                    f"{t[0]}/{t[1]}", wadvsys, wicl, inc_samples[t[1]]
                )
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(
                f"{topic[0]}/{topic[1]}", wadvsys, wicl, inc_samples[topic[1]]
            )

        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if (
            isinstance(resp, list) and all(isinstance(r, str) for r in resp)
        ) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return ([""] * len(resp) if isinstance(resp, list) else "", tl, gt)

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        # Calculate total batches including the partial last batch
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    @staticmethod
    def _escape_control_chars_in_strings(text: str) -> str:
        out = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    out.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escaped = True
                    continue
                if ch == '"':
                    out.append(ch)
                    in_string = False
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    out.append("\\r")
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                out.append(ch)
            else:
                out.append(ch)
                if ch == '"':
                    in_string = True
        return "".join(out)

    def _coerce_question_dict(self, item: Any) -> Dict[str, Any] | None:
        if isinstance(item, dict):
            return item
        if not isinstance(item, str):
            return None

        candidates: List[str] = []
        stripped = self._strip_code_fences(item)
        candidates.append(stripped)
        extracted = self._extract_first_json_object(stripped)
        if extracted:
            candidates.append(extracted)

        for candidate in candidates:
            fixed = self._escape_control_chars_in_strings(candidate)
            try:
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _normalize_question_schema(q: Dict[str, Any]) -> Dict[str, Any] | None:
        required_keys = ["topic", "question", "choices", "answer", "explanation"]
        if not all(k in q for k in required_keys):
            return None
        if not isinstance(q["topic"], str) or not isinstance(q["question"], str):
            return None
        if not isinstance(q["explanation"], str):
            return None
        choices = QuestioningAgent._normalize_choices_with_labels(q["choices"])
        if choices is None:
            return None

        ans = str(q["answer"]).strip().upper()
        if ans and ans[0] in "ABCD":
            ans = ans[0]
        if ans not in "ABCD":
            return None

        return {
            "topic": q["topic"].strip(),
            "question": q["question"].strip(),
            "choices": [c.strip() for c in choices],
            "answer": ans,
            "explanation": q["explanation"].strip(),
        }

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(
        self, questions: List[str | Dict[str, str | Any]]
    ) -> List[Dict[str, str | Any]]:
        def basic_checks(q2: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["topic", "question", "choices", "answer"]
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(
                    isinstance(choice, str)
                    and len(choice) > 2
                    and choice[0].upper() in "ABCD"
                    for choice in q2["choices"]
                )
                if (
                    isinstance(q2["choices"], list)
                    and len(q2["choices"]) == 4
                    and checks
                ):
                    # check answer format
                    # Check token length
                    check_len = sum(
                        self.count_tokens_q(q2[k]) for k in ["question", "answer"]
                    )
                    check_len += (
                        sum(self.count_tokens_q(choice) for choice in q2["choices"])
                        - 15
                    )
                    if check_len < 130:
                        if (
                            check_len
                            + self.count_tokens_q(q2.get("explanation", "None"))
                            <= 1024
                        ):
                            # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                            if isinstance(q2["answer"], str):
                                return True
            return False

        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()

    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ) -> List[str]:
        """Populate topics randomly to generate num_questions number of topics"""
        if not isinstance(topics, dict):
            raise ValueError(
                "Topics must be a dictionary with topic names as keys and lists of subtopics as values."
            )

        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")

        selected_topics = random.choices(all_subtopics, k=num_questions)
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "r") as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples


# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent."
    )
    argparser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Total number of questions to generate.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/questions.json",
        help="Output file name to save the generated questions.",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for generating questions."
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(
                f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n\n"
            )
        print("\n" + "+" * 50 + "\n")

    # Coerce each generated item into a valid JSON object as robustly as possible.
    ques = []
    for q in question:
        q_obj = agent._coerce_question_dict(q)
        if q_obj is not None:
            normalized = agent._normalize_question_schema(q_obj)
            if normalized is not None:
                ques.append(normalized)
            continue

        print(f"Invalid JSON format in question: {q}")
        prompt = (
            "Extract ONLY topic, question, choices, answer, explanation as valid JSON.\n"
            "Output JSON only, no markdown/code fences.\n\n"
            "Input:\n{}\n\n"
            "Required schema:\n"
            "{{\n"
            '  "topic": "...",\n'
            '  "question": "...",\n'
            '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            '  "answer": "A|B|C|D",\n'
            '  "explanation": "..."\n'
            "}}"
        )
        reflected, _, _ = agent.agent.generate_response(
            prompt.format(q),
            "You are an expert JSON extractor.",
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
        )
        reflected_obj = agent._coerce_question_dict(reflected)
        if reflected_obj is not None:
            normalized = agent._normalize_question_schema(reflected_obj)
            if normalized is not None:
                ques.append(normalized)
        else:
            print("Skipping an item that could not be repaired into JSON.")
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace(
        "questions.json", "filtered_questions.json"
    )
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
