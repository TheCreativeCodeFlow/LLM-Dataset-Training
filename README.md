# DSA Tutor: Production Training & Inference Pipeline

Welcome to the **DSA Tutor** repository. This project is a complete, production-grade system designed to train, evaluate, and serve a fine-tuned Data Structures and Algorithms (DSA) tutoring model. 

The system utilizes **microsoft/Phi-3-mini-4k-instruct** as the base LLM, augmented by a custom-trained **LoRA PEFT Adapter** located at `models/adapters/dsa_tutor_v1`.

---

## 1. Local Setup and Prerequisites

### Hardware Requirements
- **RAM**: Minimum 16 GB (The model in 16-bit precision consumes ~7.3 GB of memory during inference).
- **GPU**: CUDA-enabled GPU is optional. The engine automatically falls back to CPU execution if no GPU is detected.

### Installation
Clone the repository and initialize the Python virtual environment:

```powershell
# Clone the repository
git clone https://github.com/TheCreativeCodeFlow/LLM-Dataset-Training.git
cd LLM-Dataset-Training

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Configuration (`configs/inference.yaml`)

The production inference stack is configured dynamically using `configs/inference.yaml`. 

```yaml
model:
  base_model: "microsoft/Phi-3-mini-4k-instruct"
  adapter_path: "./models/adapters/dsa_tutor_v1"
  torch_dtype: "bfloat16" # CPU memory optimization

inference:
  max_new_tokens: 512
  temperature: 0.3
  top_p: 0.95
  top_k: 50
  do_sample: true
  stop_sequences:
    - "<|end|>"
    - "<|im_end|>"
    - "</s>"

server:
  host: "127.0.0.1"
  port: 8000
  workers: 1
```

---

## 3. Running the Server Locally

To start the FastAPI production server:

```powershell
# Make sure virtual environment is active
& .venv\Scripts\python.exe scripts/serve.py
```

Upon startup, the server will load the base model, bind the LoRA adapter, validate architectural dimensions, run a warmup inference pass, and start the Uvicorn web server at `http://127.0.0.1:8000`.

---

## 4. API Key Authentication and Extraction

To prevent unauthorized access, all POST endpoints are protected by an API Key middleware that checks the incoming `X-API-Key` header.

### Setting Your API Key
You can configure your custom production API Key by setting the `DSA_TUTOR_API_KEY` environment variable before running the server:

```powershell
# On Windows PowerShell:
$env:DSA_TUTOR_API_KEY="my_secret_dsa_tutor_key_98765"

# On Linux/macOS:
export DSA_TUTOR_API_KEY="my_secret_dsa_tutor_key_98765"
```

*Note: If the `DSA_TUTOR_API_KEY` environment variable is not defined, the server defaults to using `dsa_tutor_prod_secure_key_2026` as the fallback secure key.*

---

## 5. Integrating with Another Project

To use the running DSA Tutor API in another project (e.g. a frontend dashboard, a backend service, or a Chrome extension), send a request with the `X-API-Key` header.

### A. Python Client Example
```python
import requests

url = "http://127.0.0.1:8000/chat"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "my_secret_dsa_tutor_key_98765"  # Replace with your actual key
}
payload = {
    "session_id": "student_session_001",
    "query": "Explain how search operations work in a BST"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("Mode:", data["tutor_mode"])
    print("Topic:", data["topic"])
    print("Response:", data["response"])
else:
    print(f"Error {response.status_code}: {response.text}")
```

### B. JavaScript / Node.js Fetch Example
```javascript
const url = 'http://127.0.0.1:8000/chat';
const apiKey = 'my_secret_dsa_tutor_key_98765'; // Replace with your actual key

const requestData = {
  session_id: 'student_session_001',
  query: 'Give me a hint on how to reverse a linked list'
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': apiKey
  },
  body: JSON.stringify(requestData)
})
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    console.log('Tutor Response:', data.response);
  })
  .catch(error => {
    console.error('API Call Failed:', error);
  });
```

### C. cURL CLI Example
```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -H "X-API-Key: my_secret_dsa_tutor_key_98765" \
     -d '{
       "session_id": "session_demo",
       "query": "What is the time complexity of bubble sort?"
     }'
```

---

## 6. Manual Evaluation CLI Environment

You can manually test, interact with, and score the model using the built-in CLI application:

```powershell
& .venv\Scripts\python.exe scripts/evaluate_interactive.py
```

### CLI Features:
- **Interactive Chat**: Keeps full session history and streams responses in real-time.
- **Tutor Mode Commands**: Switch active tutor personas on the fly:
  - `/beginner` (Beginner Tutor)
  - `/interview` (Technical Coach)
  - `/debug` (Debugging Mentor)
  - `/review` (Code Reviewer)
  - `/complexity` (Complexity Analyst)
  - `/hint` (Hint Generator)
- **Scoring**: Prompts you to rate each answer from 1 to 5 on educational value, technical correctness, logical consistency, etc.
- **Failure Diagnostics**: Automatically collects and logs failures in `dataset/failures/manual_failures.jsonl` if rating <= 2.
- **Session Reporting**: Generates a detailed Markdown report at `evaluation/manual_tests/session_report.md` upon exit.

---

## 7. Automated Regression Testing

To verify the API performance metrics, routing accuracy, and safety constraints on a suite of 100 regression queries:

```powershell
& .venv\Scripts\python.exe tests/test_tutor_api.py
```
- Results are saved to `logs/tutor_regression_report.json`.