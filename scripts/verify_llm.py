#!/usr/bin/env python3
"""
IO Matrix Verification for GitPry LLM Integration
"""

import sys
from gitpry.llm.client import stream_ollama
from gitpry.llm.prompts import SYSTEM_PROMPT, build_user_prompt

if __name__ == "__main__":
    print("--- Test 1: Testing Local Ollama Connection ---")
    
    mock_context = (
        "1a2b3c4d | Yifan Ke | 2026-02-28 10:00:00 | fix(db): bypass slow query in user auth\n"
        "5e6f7g8h | John Doe | 2026-02-27 09:00:00 | feat: add initial user authentication"
    )
    question = "Who bypassed the slow query and in which commit?"
    
    prompt = build_user_prompt(mock_context, question)
    
    print(f"Targeting default model (qwen2.5-coder:7b). Ensure Ollama is running.\n")
    print("Querying...\n--- LLM Response ---")
    
    chunks_received = 0
    generator = stream_ollama(prompt=prompt, system=SYSTEM_PROMPT, model="qwen2.5-coder:7b")
    
    if generator:
        for chunk in generator:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            chunks_received += 1
        
        print(f"\n\n✓ Received {chunks_received} chunks successfully.")
    else:
        print("\n✗ Failed to get a response. Expected if Ollama is not installed or model is missing.")
