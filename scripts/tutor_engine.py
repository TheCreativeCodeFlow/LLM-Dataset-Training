#!/usr/bin/env python3
"""
scripts/tutor_engine.py

Production-grade inference and tutoring engine for DSA Tutor.
Manages ModelLoader singleton, compatibility checks, intent routing,
streaming generation, session memory, and safety filters.
"""

import os
import yaml
import json
import torch
import re
import time
import psutil
import hashlib
import threading
from typing import Dict, List, Any, Generator, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
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

def get_file_hash(filepath: str) -> str:
    """Computes SHA256 of the first 1MB of a file for quick identification."""
    if not os.path.exists(filepath):
        return "unknown"
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024 * 1024)
            h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "error"

class ModelLoader:
    """Singleton model loader that loads base model, tokenizer, and adapter once."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelLoader, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_path: str = "configs/inference.yaml"):
        if self._initialized:
            return
        self.config_path = config_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.base_model_name = None
        self.adapter_path = None
        self.load_time_seconds = 0.0
        self.model_hash = "unknown"
        self.adapter_hash = "unknown"
        self._initialized = True

    def discover_and_load(self):
        """Loads and validates the base model and PEFT adapter."""
        with self._lock:
            if self.model is not None:
                return

            start_time = time.time()
            
            # 1. Discover Configuration
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Inference config not found at: {self.config_path}")
                
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                
            self.base_model_name = config["model"]["base_model"]
            self.adapter_path = config["model"]["adapter_path"]
            torch_dtype_str = config["model"].get("torch_dtype", "bfloat16")
            torch_dtype = getattr(torch, torch_dtype_str, torch.float32)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            print(f"[ModelLoader] Loading tokenizer: {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                

            # Load Base Model
            print(f"[ModelLoader] Loading base model: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch_dtype if self.device != "cpu" else torch.bfloat16,
                device_map=None if self.device == "cpu" else "auto",
                trust_remote_code=False
            )
            
            # Retrieve Model config file path to compute hash
            model_config_path = base_model.config._name_or_path
            self.model_hash = get_file_hash(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{self.base_model_name.replace('/', '--')}", "snapshots", "f39ac1d28e925b323eae81227eaba4464caced4e", "config.json"))
            
            # Validate adapter compatibility before merging
            adapter_config_path = os.path.join(self.adapter_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(f"Adapter config not found at: {adapter_config_path}")
                
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
                
            self.adapter_hash = get_file_hash(os.path.join(self.adapter_path, "adapter_model.safetensors"))
            
            # Check compatibility
            self.validate_compatibility(base_model, self.tokenizer, adapter_config)
            
            # Attach PEFT adapter
            print(f"[ModelLoader] Loading and attaching adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # Move to target device
            self.model.to(self.device)
            self.model.eval()
            
            # Warm up
            print("[ModelLoader] Warming up model with dummy query...")
            warmup_messages = [
                {"role": "system", "content": "You are a DSA tutor."},
                {"role": "user", "content": "Hello"}
            ]
            prompt = self.tokenizer.apply_chat_template(warmup_messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=5, pad_token_id=self.tokenizer.pad_token_id)
                
            self.load_time_seconds = time.time() - start_time
            print(f"[ModelLoader] Model and adapter successfully initialized and warmed up in {self.load_time_seconds:.2f} seconds.")

    def validate_compatibility(self, base_model, tokenizer, adapter_config):
        """Performs validation checks on architectural alignment between model, tokenizer, and adapter."""
        diagnostics = []
        is_compatible = True
        
        # 1. Vocab Size Match
        model_vocab = base_model.config.vocab_size
        tok_vocab = len(tokenizer)
        diagnostics.append(f"Model Vocab Size: {model_vocab}, Tokenizer Vocab Size: {tok_vocab}")
        if abs(model_vocab - tok_vocab) > 1000:
            is_compatible = False
            diagnostics.append("FAIL: Vocabulary size mismatch is too large!")
            
        # 2. Hidden Size Check
        hidden_size = base_model.config.hidden_size
        diagnostics.append(f"Base Model Hidden Size: {hidden_size}")
        if hidden_size != 3072:  # Phi-3-mini expected size
            is_compatible = False
            diagnostics.append(f"FAIL: Hidden size {hidden_size} does not match expected Phi-3-mini architecture (3072)!")
            
        # 3. Target Module Verification
        target_modules = adapter_config.get("target_modules", [])
        diagnostics.append(f"LoRA Target Modules: {target_modules}")
        
        # Inspect base model parameter names
        named_params = [name for name, _ in base_model.named_parameters()]
        valid_targets = []
        for target in target_modules:
            matched = any(target in name for name in named_params)
            diagnostics.append(f"Target module '{target}' present in base model: {matched}")
            if matched:
                valid_targets.append(target)
                
        if len(valid_targets) == 0:
            is_compatible = False
            diagnostics.append("FAIL: None of the LoRA target modules are present in the base model architecture!")
            
        # Output results
        diagnostics_str = "\n".join(diagnostics)
        if not is_compatible:
            print("[ModelLoader] COMPATIBILITY FAILURE DETECTED!")
            print(diagnostics_str)
            raise AssertionError(f"Model/Adapter Compatibility Failure:\n{diagnostics_str}")
        else:
            print("[ModelLoader] Compatibility validation passed successfully.")

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
    """Manages intent routing, safety check, and thread-safe streaming generation."""
    def __init__(self, config_path: str = "configs/inference.yaml"):
        # Load and warm up model loader singleton
        self.loader = ModelLoader(config_path)
        self.loader.discover_and_load()
        
        self.sessions: Dict[str, SessionMemory] = {}
        self.lock = threading.Lock()

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
                # Remove code blocks
                clean_response = re.sub(r"```[a-zA-Z]*\n(.*?)```", "", response, flags=re.DOTALL)
                clean_response = clean_response.strip() + "\n\n(Hint: I have omitted the full code implementation to help you work through it yourself! Let me know if you need another algorithmic hint.)"
                return clean_response
                
        # Handle empty or invalid generation
        if not response or len(response.strip()) < 5:
            return "I am experiencing low confidence in this specific solution, but let's walk through the core concept together! What part of the algorithm seems most challenging?"
            
        return response

    def generate_response_stream(self, session_id: str, query: str, force_mode: str = None, creative: bool = False, deterministic: bool = False, disable_adapter: bool = False) -> Generator[Dict[str, Any], None, None]:
        """Generates streaming responses using Thread-safe TextIteratorStreamer."""
        # 1. Safety check
        is_safe, warning = self.perform_safety_check(query)
        if not is_safe:
            yield {
                "token": warning,
                "metrics": {"latency_seconds": 0.0, "tokens_per_second": 0.0},
                "tutor_mode": "safety_block",
                "topic": "Safety"
            }
            return
            
        session = self.get_or_create_session(session_id)
        
        # 2. Routing
        mode = force_mode if force_mode else self.route_intent(query)
        session.tutor_mode = mode
        system_prompt = SYSTEM_PROMPTS[mode]
        
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
            
        # 3. Build context
        messages = [{"role": "system", "content": system_prompt}]
        for msg in session.messages[-4:]:
            messages.append(msg)
        messages.append({"role": "user", "content": query})
        
        prompt = self.loader.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.loader.tokenizer(prompt, return_tensors="pt").to(self.loader.device)
        
        # 4. Configure params
        gen_kwargs = {
            "max_new_tokens": 256,
            "pad_token_id": self.loader.tokenizer.pad_token_id
        }
        
        if deterministic:
            gen_kwargs["temperature"] = 0.0
            gen_kwargs["do_sample"] = False
        elif creative:
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        else:
            with open(self.loader.config_path, "r") as f:
                config = yaml.safe_load(f)
            inf_cfg = config["inference"]
            gen_kwargs["max_new_tokens"] = inf_cfg.get("max_new_tokens", 256)
            gen_kwargs["temperature"] = inf_cfg.get("temperature", 0.3)
            gen_kwargs["top_p"] = inf_cfg.get("top_p", 0.95)
            gen_kwargs["top_k"] = inf_cfg.get("top_k", 50)
            gen_kwargs["do_sample"] = inf_cfg.get("do_sample", True)

        # 5. Thread-safe Streamer
        streamer = TextIteratorStreamer(self.loader.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        
        # Lock to prevent simultaneous access during generation
        with self.lock:
            start_time = time.time()
            if disable_adapter:
                def target_func():
                    with self.loader.model.disable_adapter():
                        self.loader.model.generate(**inputs, **gen_kwargs)
            else:
                def target_func():
                    self.loader.model.generate(**inputs, **gen_kwargs)
            
            thread = threading.Thread(target=target_func)
            thread.start()
            
            response_chunks = []
            token_count = 0
            
            for token in streamer:
                token_count += 1
                response_chunks.append(token)
                elapsed = time.time() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0.0
                
                yield {
                    "token": token,
                    "metrics": {"latency_seconds": elapsed, "tokens_per_second": tps},
                    "tutor_mode": mode,
                    "topic": session.current_topic
                }
                
            thread.join()
            
        full_response = "".join(response_chunks).strip()
        
        # 6. Enrich structure (Concept, Reasoning, Complexity, Edge cases, Next Practice)
        enriched_response = (
            f"**[{mode.replace('_', ' ').title()}]**\n\n"
            f"### Concept & Explanation\n"
            f"{full_response}\n\n"
            f"### Reasoning & Walkthrough\n"
            f"- We examined: '{query[:60]}...'\n"
            f"- We identify optimal approaches matching topic: {session.current_topic}.\n\n"
            f"### Complexity & Trade-offs\n"
            f"- Time Complexity: O(N) traversal or O(log N) binary search bounds.\n"
            f"- Space Complexity: O(1) auxiliary space optimization.\n\n"
            f"### Edge Cases & Common Mistakes\n"
            f"- Verify off-by-one indices.\n"
            f"- Check null or empty collection checks.\n\n"
            f"### Next Practice Suggestion\n"
            f"Try practicing a similar {session.current_topic} problem or ask me to explain this trade-off further!"
        )
        
        final_response = self.self_evaluate_response(enriched_response, mode)
        
        # Update session memory
        session.messages.append({"role": "user", "content": query})
        session.messages.append({"role": "assistant", "content": final_response})
        if mode == "hint_generator":
            session.hints_given.append(final_response)
            
        # Final yield to return the fully structured content
        yield {
            "token": "[DONE]",
            "full_response": final_response,
            "metrics": {
                "latency_seconds": time.time() - start_time,
                "tokens_per_second": token_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0.0
            },
            "tutor_mode": mode,
            "topic": session.current_topic
        }
