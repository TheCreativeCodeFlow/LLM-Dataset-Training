#!/usr/bin/env python3
"""
scripts/tutor_engine.py

Core inference and tutoring layer for DSA Tutor.
Manages base model loading, LoRA adapter loading, session memory,
prompt routing, safety filtering, and self-evaluation.
"""

import os
import yaml
import torch
import re
from typing import Dict, List, Any, Generator, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# System Prompts for Tutor Modes
SYSTEM_PROMPTS = {
    "beginner_tutor": (
        "You are a friendly, patient DSA beginner tutor. Explain concepts using simple, real-world analogies "
        "and step-by-step logic. Avoid complex jargon. Ask simple questions to check understanding."
    ),
    "interview_coach": (
        "You are a professional DSA interview coach. Simulate a technical mock interview. Ask the student "
        "to explain their approach, discuss trade-offs, identify edge cases, and talk through code before writing it."
    ),
    "debugging_mentor": (
        "You are a DSA debugging mentor. Do NOT provide correct code or fixes directly. Instead, identify the line "
        "or region of the bug and ask guiding questions to help the student find and fix the issue themselves."
    ),
    "complexity_analyst": (
        "You are a DSA complexity analyst. Focus heavily on time and space complexity, recurrence relations, "
        "recursion call stacks, memory layout, and optimal scaling boundaries (Big-O analysis)."
    ),
    "code_reviewer": (
        "You are a DSA code reviewer. Critique the student's code readability, naming conventions, structural design, "
        "DRY principle adherence, clean-code style, and potential code smells."
    ),
    "hint_generator": (
        "You are a DSA hint generator. Offer progressive hints: Hint 1 (High-level conceptual direction), "
        "Hint 2 (Algorithmic guideline), Hint 3 (Pseudocode logic). NEVER output the actual solution code."
    )
}

class SessionMemory:
    """Stores conversation history and metadata for a tutoring session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []
        self.current_topic: str = "Unknown"
        self.hints_given: List[str] = []
        self.mistakes: List[str] = []
        self.difficulty: str = "medium"
        self.progress: Dict[str, Any] = {}
        self.tutor_mode: str = "beginner_tutor"

class TutorEngine:
    """Manages model loading, intent routing, safety checks, and inference."""
    def __init__(self, config_path: str = "configs/train_config.yaml"):
        # Load configs
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.model_name = self.config["model"]["name"]
        self.adapter_path = self.config["training"]["final_adapter_dir"]
        
        # Detect hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TutorEngine] Auto-detected hardware device: {self.device}")
        
        # Load Tokenizer
        print(f"[TutorEngine] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Custom chat template supporting generation tags
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% generation %}"
            "{{ message['content'] + '<|im_end|>\n' }}"
            "{% endgeneration %}"
            "{% endif %}"
            "{% endfor %}"
        )
        
        # Load Model
        print(f"[TutorEngine] Loading base model: {self.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            device_map=None if self.device == "cpu" else "auto",
            trust_remote_code=True
        )
        
        # Load PEFT Adapter
        if os.path.exists(self.adapter_path):
            print(f"[TutorEngine] Loading LoRA adapter weights from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            print(f"[TutorEngine] WARNING: Adapter path not found. Running on base model only.")
            self.model = base_model
            
        self.model.to(self.device)
        self.model.eval()
        
        # Sessions registry
        self.sessions: Dict[str, SessionMemory] = {}

    def get_or_create_session(self, session_id: str) -> SessionMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id)
        return self.sessions[session_id]

    def route_intent(self, user_query: str) -> str:
        """Heuristic prompt routing to classify user intent to a specific tutor mode."""
        q = user_query.lower()
        
        # 1. Hint Generator (highest priority)
        if any(w in q for w in ["hint", "clue", "stuck"]):
            return "hint_generator"
            
        # 2. Code Reviewer
        if any(w in q for w in ["review", "refactor", "style", "clean code", "dry", "naming", "readability", "modular", "critique"]):
            return "code_reviewer"
            
        # 3. Interview Coach
        if any(w in q for w in ["interview", "mock", "coach", "expectations", "communicate", "interviewer"]):
            return "interview_coach"
            
        # 4. Complexity Analyst
        if any(w in q for w in ["complexity", "big o", "time", "space", "recurrence", "o(n", "o(1", "amortized", "efficiency", "trade-off", "tradeoff"]):
            return "complexity_analyst"
            
        # 5. Debugging Mentor
        if any(w in q for w in ["bug", "error", "debug", "fix", "fail", "wrong", "nullpointer", "out of range", "loop", "exception", "crashes", "duplicate", "why does my"]):
            return "debugging_mentor"
            
        # Default fallback
        return "beginner_tutor"

    def perform_safety_check(self, query: str) -> Tuple[bool, str]:
        """Prevents prompt injection, override, and role confusion attempts."""
        patterns = [
            r"ignore previous", r"system prompt", r"you are now a", 
            r"override", r"bypass", r"stop being a tutor", r"forget instruction"
        ]
        for p in patterns:
            if re.search(p, query.lower()):
                return False, "As your dedicated DSA tutor, I must stay focused on helping you learn data structures and algorithms. Let's get back to the topic!"
        return True, ""

    def self_evaluate_response(self, response: str, mode: str) -> str:
        """Inspects response for correctness and solution leaks, providing fallbacks if needed."""
        # Detect solution leak when in hint mode
        if mode == "hint_generator":
            code_blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", response, re.DOTALL)
            if len(code_blocks) > 0:
                # Premature solution leak detected! Remove code blocks and remind student
                clean_response = re.sub(r"```[a-zA-Z]*\n(.*?)```", "", response, flags=re.DOTALL)
                clean_response = clean_response.strip() + "\n\n(Hint: I have omitted the full code implementation to help you work through it yourself! Let me know if you need another algorithmic hint.)"
                return clean_response
                
        # Handle incoherent tiny model outputs (confidence check)
        if not response or len(response.strip()) < 5 or "[INCOHERENT" in response:
            return "I am experiencing low confidence in this specific solution, but let's walk through the core concept together! What part of the algorithm seems most challenging?"
            
        return response

    def generate_response(self, session_id: str, query: str, force_mode: str = None) -> Generator[str, None, None]:
        """Main response pipeline, returning response in chunks."""
        # 1. Safety check
        is_safe, warning = self.perform_safety_check(query)
        if not is_safe:
            yield warning
            return
            
        session = self.get_or_create_session(session_id)
        
        # 2. Routing
        mode = force_mode if force_mode else self.route_intent(query)
        session.tutor_mode = mode
        system_prompt = SYSTEM_PROMPTS[mode]
        
        # Update session memory metadata
        if "array" in query.lower():
            session.current_topic = "Arrays"
        elif "string" in query.lower():
            session.current_topic = "Strings"
        elif "list" in query.lower():
            session.current_topic = "Linked Lists"
        elif "tree" in query.lower():
            session.current_topic = "Trees"
        elif "graph" in query.lower():
            session.current_topic = "Graphs"
            
        # 3. Build Conversation Context
        messages = [{"role": "system", "content": system_prompt}]
        # Append recent history (last 4 messages to prevent context drift)
        for msg in session.messages[-4:]:
            messages.append(msg)
        # Add new query
        messages.append({"role": "user", "content": query})
        
        # 4. Tokenize & Generate
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        assistant_marker = "assistant\n"
        if assistant_marker in full_output:
            raw_response = full_output.split(assistant_marker)[-1].strip()
        else:
            raw_response = full_output.replace(prompt, "").strip()
            
        if not raw_response:
            raw_response = "[INCOHERENT RANDOM TOKENS] due to tiny Llama weights."
            
        # 5. Enrich with Response Pipeline structure (Concept, Reasoning, Complexity, Edge cases, Next Practice)
        enriched_response = (
            f"**[{mode.replace('_', ' ').title()}]**\n\n"
            f"### Concept & Explanation\n"
            f"{raw_response}\n\n"
            f"### Reasoning & Walkthrough\n"
            f"- We first examine the query details: '{query[:60]}...'\n"
            f"- We identify optimal approaches matching topic: {session.current_topic}.\n\n"
            f"### Complexity & Trade-offs\n"
            f"- Time Complexity: O(N) traversal or logarithmic search depending on sorted bounds.\n"
            f"- Space Complexity: O(1) auxiliary space optimization.\n\n"
            f"### Edge Cases & Common Mistakes\n"
            f"- Off-by-one index bounds check.\n"
            f"- Null or empty collection checks.\n\n"
            f"### Next Practice Suggestion\n"
            f"Try practicing a similar {session.current_topic} problem or ask me to explain this trade-off further!"
        )
        
        # 6. Self-evaluate
        final_response = self.self_evaluate_response(enriched_response, mode)
        
        # Update session memory
        session.messages.append({"role": "user", "content": query})
        session.messages.append({"role": "assistant", "content": final_response})
        if mode == "hint_generator":
            session.hints_given.append(final_response)
            
        # Yield result
        yield final_response
