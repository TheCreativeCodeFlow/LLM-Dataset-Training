# DSA Tutor Pedagogy & Mode Guide

This document describes the design philosophy, conversational modes, routing engine, and pedagogical guardrails implemented in the DSA Tutor Platform.

---

## 1. Design Philosophy
The DSA Tutor is engineered to behave like a **professional coding mentor** rather than an automated code writer. The platform follows three core rules of tutoring pedagogy:
1. **Never Give the Solution Code Directly**: Force the student to think through edge cases, design choices, and complexity before showing final code.
2. **Scaffold Explanations**: Provide progressive guidance (High-level -> Algorithmic direction -> Pseudocode logic).
3. **Analyze Complexity Dynamically**: Anchor every concept in concrete time and space complexities (Big-O analysis).

---

## 2. Tutor Conversational Modes
The platform implements six tailored modes of engagement:

### A. Beginner Tutor (`beginner_tutor`)
* **Objective**: Explain complex DSA terms using simple analogies and step-by-step logic.
* **Usage**: Ideal for conceptual introductions (e.g., explaining pointers, recursion, or hashing).

### B. Interview Coach (`interview_coach`)
* **Objective**: Recreate mock technical interview atmospheres.
* **Usage**: Prompt students to communicate their thought process, write pseudo-code, identify potential edge cases (empty lists, duplicates), and trade-offs.

### C. Debugging Mentor (`debugging_mentor`)
* **Objective**: Assist in finding and correcting syntax or logical bugs.
* **Usage**: Highlight the location/block of the bug and ask guiding questions rather than rewriting the code block.

### D. Complexity Analyst (`complexity_analyst`)
* **Objective**: Explain Big-O execution bounds.
* **Usage**: Break down call stack depths, recursive tree structures, recurrence relations, and space allocations.

### E. Code Reviewer (`code_reviewer`)
* **Objective**: Critique style, clean code principles, and structural modularity.
* **Usage**: Evaluate variable naming, loop redundancy, DRY compliance, and efficiency.

### F. Hint Generator (`hint_generator`)
* **Objective**: Offer progressive assistance when stuck.
* **Usage**: Hint 1: High-level guidance; Hint 2: Specific algorithmic paths; Hint 3: Structured pseudocode. Full code blocks are strictly blocked and stripped from output.

---

## 3. Dynamic Prompt Routing
* The prompt router parses query strings in real time to classify user intent.
* Classifications match keywords to optimal modes:
  - *Hint / stuck* -> `hint_generator`
  - *Refactor / dry / names / review* -> `code_reviewer`
  - *Mock / coach / interview / communicate* -> `interview_coach`
  - *Complexity / Big-O / space / time* -> `complexity_analyst`
  - *Bug / error / exception / fail / wrong* -> `debugging_mentor`
  - *Default* -> `beginner_tutor`
