#!/usr/bin/env python3
"""
scripts/code_analyst.py

Local static analysis engine supporting Python, Java, C++, and JavaScript.
Detects syntax checks, logical loops, estimates complexity, and identifies edge cases.
"""

import re
from typing import Dict, Any, List

class CodeAnalyst:
    def __init__(self):
        pass

    def detect_language(self, code: str) -> str:
        """Heuristically detects programming language of code snippet."""
        if "def " in code or "import sys" in code or ("print(" in code and "class " not in code and ";" not in code):
            return "python"
        if "public class" in code or "System.out.print" in code or "String[] args" in code:
            return "java"
        if "#include" in code or "std::" in code or "cout <<" in code or "int main(" in code:
            return "cpp"
        if "const " in code or "let " in code or "function " in code or "console.log" in code:
            return "javascript"
        # Fallback default
        return "python"

    def analyze(self, code: str) -> Dict[str, Any]:
        """Runs static checks on code and returns tutoring guidance details."""
        lang = self.detect_language(code)
        issues = []
        logical_bugs = []
        edge_cases = []
        
        # 1. Syntax Check
        if lang == "python":
            # Check indentation basics or missing colons
            lines = code.splitlines()
            for idx, line in enumerate(lines):
                if any(kw in line for kw in ["def ", "if ", "for ", "while ", "class "]) and not line.strip().endswith(":"):
                    issues.append(f"Line {idx+1}: Missing colon ':' at the end of block statement.")
        else:
            # Check semicolons or brackets
            open_braces = code.count("{")
            close_braces = code.count("}")
            if open_braces != close_braces:
                issues.append(f"Unbalanced braces found: {open_braces} opening vs {close_braces} closing braces.")
            # Check missing semicolons
            lines = code.splitlines()
            for idx, line in enumerate(lines):
                strip_l = line.strip()
                if strip_l and not strip_l.endswith(";") and not strip_l.endswith("{") and not strip_l.endswith("}") and not strip_l.startswith("//") and not strip_l.startswith("#"):
                    if any(w in strip_l for w in ["return", "int ", "double ", "let ", "const ", "="]):
                        issues.append(f"Line {idx+1}: Potential missing semicolon ';'.")

        # 2. Logical Bugs Detection
        if "while True" in code or "while(true)" in code:
            if "break" not in code:
                logical_bugs.append("Potential infinite loop detected. A 'while (true)' block exists without a visible 'break' statement.")
        if "/ 0" in code or "/0" in code:
            logical_bugs.append("Division by zero operation detected.")
        if "recursion" in code.lower() or "def solve" in code or "dfs(" in code:
            if "if" not in code:
                logical_bugs.append("Recursion detected but no visible base case conditional ('if') was identified. Verify termination.")

        # 3. Complexity Estimation
        time_comp = "O(1)"
        space_comp = "O(1)"
        
        # Count loop structures
        loops = len(re.findall(r'\b(for|while)\b', code))
        if loops == 1:
            time_comp = "O(N)"
        elif loops >= 2:
            # Check nested loops heuristically
            nested = False
            lines = [l.strip() for l in code.splitlines() if l.strip()]
            for i in range(len(lines) - 1):
                if lines[i].startswith("for") or lines[i].startswith("while"):
                    if lines[i+1].startswith("for") or lines[i+1].startswith("while"):
                        nested = True
            time_comp = "O(N^2)" if nested else "O(N)"
            
        if "binary_search" in code.lower() or "mid =" in code or "low <=" in code:
            time_comp = "O(log N)"
            
        # Space complexity guess
        if "new int[" in code or "append(" in code or "vector<" in code or "[]" in code:
            space_comp = "O(N)"

        # 4. Edge Cases Checklist
        if "arr" in code.lower() or "nums" in code.lower() or "list" in code.lower():
            edge_cases.append("Empty collection or single element arrays.")
            edge_cases.append("Null pointer dereference checks.")
        if "low + high" in code:
            edge_cases.append("Integer overflow bounds (prefer: low + (high - low) / 2).")

        # 5. Tutor Feedback Generation
        feedback = []
        if issues:
            feedback.append(f"**Syntax Tips**: We noticed a few styling/syntax issues: {'; '.join(issues)}")
        if logical_bugs:
            feedback.append(f"**Logic Warning**: {'; '.join(logical_bugs)}")
        else:
            feedback.append("**Logic check**: No obvious logical loops or crash points found. Looks structurally stable!")
            
        feedback.append(f"**Complexity Profile**: Estimating Time Complexity as *{time_comp}* and Auxiliary Space Complexity as *{space_comp}*.")
        
        if edge_cases:
            feedback.append(f"**Edge Cases to Watch**: Be sure to test your code against: {', '.join(edge_cases)}")

        return {
            "language": lang,
            "syntax_issues": issues,
            "logical_bugs": logical_bugs,
            "time_complexity": time_comp,
            "space_complexity": space_comp,
            "edge_cases": edge_cases,
            "tutor_feedback": "\n\n".join(feedback)
        }

if __name__ == "__main__":
    print("=== Testing Code Analyst ===")
    analyst = CodeAnalyst()
    
    test_python = """
def find_sum(arr):
    total = 0
    for num in arr
        total += num
    return total
"""
    report = analyst.analyze(test_python)
    print(f"Detected Language: {report['language']}")
    print(report["tutor_feedback"])
