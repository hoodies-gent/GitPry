#!/usr/bin/env python3
"""
IO Matrix Verification for GitPry Core Git Integration
"""

from gitpry.git_utils.repository import get_recent_commits, build_prompt_context
from gitpry.utils.logger import setup_logger

if __name__ == "__main__":
    # Test 1: Invalid repository path
    print("--- Test 1: Invalid Repository (Expected to fail gracefully) ---")
    invalid_commits = get_recent_commits(repo_path="/tmp")
    assert invalid_commits is None, "Should return None for invalid repos"
    print("✓ Gracefully handled invalid path.\n")

    # Test 2: Valid repository
    print("--- Test 2: Current Repository (Expected to succeed) ---")
    commits = get_recent_commits(limit=3)
    
    if commits:
        print(f"Extracted {len(commits)} commits. Raw Data:")
        for idx, c in enumerate(commits):
            print(f"  {idx + 1}. {c['hash']}: {c['message'][:30]}...")
            
        print("\n--- Test 3: Prompt Text Assembly ---")
        context_str = build_prompt_context(commits)
        print("Generated Context String:")
        print(context_str)
        print("\n✓ Valid repository processing successful.")
    else:
        print("✗ Failed to extract commits from the current repository.")
