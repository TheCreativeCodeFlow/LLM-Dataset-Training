#!/usr/bin/env python3
"""
scripts/chunk_kb.py

Reads monolithic dsa_kb.json and splits it into granular semantic chunks
with topic, type, difficulty, and keyword metadata. Saves to dsa_kb_chunks.json.
"""

import os
import json
import re

KB_PATH = "data/knowledge_base/dsa_kb.json"
OUTPUT_PATH = "data/knowledge_base/dsa_kb_chunks.json"

def clean_keywords(text: str) -> list:
    words = re.findall(r'[a-z]{3,}', text.lower())
    stopwords = {"and", "the", "for", "with", "this", "that", "from", "are", "have", "you", "can"}
    return sorted(list(set(w for w in words if w not in stopwords)))[:10]

def determine_difficulty(topic: str) -> str:
    hard_topics = {"AVL", "Trie", "Graphs", "Union Find", "Dynamic Programming", "Backtracking"}
    easy_topics = {"Arrays", "Strings", "Stack", "Queue", "Prefix Sum"}
    if topic in hard_topics:
        return "hard"
    if topic in easy_topics:
        return "easy"
    return "medium"

def main():
    if not os.path.exists(KB_PATH):
        print(f"Error: Base knowledge base not found at {KB_PATH}")
        return
        
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_items = json.load(f)
        
    chunks = []
    
    for item in kb_items:
        topic = item["topic"]
        diff = determine_difficulty(topic)
        
        # 1. Concept Chunk
        concept_content = f"Topic: {topic}. Concept: {item['concept']}. Intuition: {item['intuition']}"
        chunks.append({
            "chunk_id": f"{topic.lower().replace(' ', '_')}_concept",
            "topic": topic,
            "difficulty": diff,
            "type": "concept",
            "keywords": sorted(list(set([topic.lower(), "concept", "intuition"] + clean_keywords(item["concept"])))),
            "content": concept_content
        })
        
        # 2. Complexity Chunk
        time_comp = item["complexities"].get("time", item["complexities"].get("search", "N/A"))
        space_comp = item["complexities"].get("space", "N/A")
        complexity_content = f"Topic: {topic} complexities. Time Complexity: {time_comp}. Space Complexity: {space_comp}."
        chunks.append({
            "chunk_id": f"{topic.lower().replace(' ', '_')}_complexity",
            "topic": topic,
            "difficulty": diff,
            "type": "complexity",
            "keywords": [topic.lower(), "complexity", "big o", "time", "space", "bounds"],
            "content": complexity_content
        })
        
        # 3. Mistakes Chunk
        mistakes_content = f"Topic: {topic} common mistakes & pitfalls: {'. '.join(item['common_mistakes'])}."
        chunks.append({
            "chunk_id": f"{topic.lower().replace(' ', '_')}_mistake",
            "topic": topic,
            "difficulty": diff,
            "type": "mistake",
            "keywords": [topic.lower(), "mistake", "pitfall", "error", "bug", "common"],
            "content": mistakes_content
        })
        
        # 4. Edge Cases Chunk
        edges_content = f"Topic: {topic} edge cases: {'. '.join(item['edge_cases'])}."
        chunks.append({
            "chunk_id": f"{topic.lower().replace(' ', '_')}_edge_case",
            "topic": topic,
            "difficulty": diff,
            "type": "edge_case",
            "keywords": [topic.lower(), "edge", "boundary", "null", "empty", "overflow"],
            "content": edges_content
        })
        
        # 5. Interview Tips Chunk
        interview_content = f"Topic: {topic} interview tips: {item['interview_tips']}. Practice Problems: {', '.join(item['practice_problems'])}."
        chunks.append({
            "chunk_id": f"{topic.lower().replace(' ', '_')}_interview",
            "topic": topic,
            "difficulty": diff,
            "type": "interview",
            "keywords": [topic.lower(), "interview", "mock", "practice", "problem", "leetcode"],
            "content": interview_content
        })
        
    # Save the chunked database
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
        
    print(f"Successfully generated {len(chunks)} chunks and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
