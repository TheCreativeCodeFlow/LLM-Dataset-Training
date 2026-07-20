# DSA Tutor API Documentation

The DSA Tutor exposes a RESTful FastAPI local server. All responses are structured to support session memory and pedagogical formatting.

---

## 1. Endpoints

### GET `/health`
* **Description**: Verifies that the FastAPI server is running and the model/adapters are loaded.
* **Response Example**:
```json
{
  "status": "ok",
  "tutor_engine_loaded": true,
  "available_modes": ["beginner_tutor", "interview_coach", "debugging_mentor", "complexity_analyst", "code_reviewer", "hint_generator"]
}
```

---

### POST `/chat`
* **Description**: General chat endpoint. Evaluates query intent, routes to optimal tutor mode, updates session history, and returns a structured response.
* **Request Schema**:
```json
{
  "session_id": "session_123",
  "query": "How does a BST insertion work?",
  "max_tokens": 256,
  "temperature": 0.3
}
```
* **Response Schema**:
```json
{
  "session_id": "session_123",
  "tutor_mode": "beginner_tutor",
  "topic": "Trees",
  "response": "**[Beginner Tutor]**\n\n### Concept & Explanation\n..."
}
```

---

### POST `/hint`
* **Description**: Forces the engine into Hint Generator mode. Restricts generation from returning complete solution code.
* **Request Schema**:
```json
{
  "session_id": "session_123",
  "query": "I am stuck on valid parentheses logic."
}
```

---

### POST `/review`
* **Description**: Forces the engine into Code Reviewer mode to critique variable naming, readability, and DRY principles.
* **Request Schema**:
```json
{
  "session_id": "session_123",
  "query": "Review: `for i in range(len(arr)): print(arr[i])`"
}
```

---

### POST `/debug`
* **Description**: Forces the engine into Debugging Mentor mode to guide the student towards resolving a code bug.
* **Request Schema**:
```json
{
  "session_id": "session_123",
  "query": "Why does my BFS traversal loop infinitely?"
}
```

---

### POST `/complexity`
* **Description**: Forces the engine into Complexity Analyst mode to evaluate Big-O bounds.
* **Request Schema**:
```json
{
  "session_id": "session_123",
  "query": "What is the complexity of merge sort?"
}
```
