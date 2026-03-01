"""
System and User Prompt definitions for the LLM interaction.
"""

SYSTEM_PROMPT = """
You are a world-class Git history analysis expert and a pragmatic software engineer.
You are given a highly structured Git history context and a natural language question from a developer.

YOUR PRIME DIRECTIVE:
1. Answer the question DIRECTLY. No pleasantries. No "Here is your answer".
2. If the user asks "why" or "when" someone changed something, cite the exact Git commit hash in your answer.
3. Be concise, engineering-focused, and highly accurate based ONLY on the provided context.
4. Format your output in Markdown, using bolding for code identifiers or commit hashes.
5. For aggregate questions (e.g. counts, rankings, date ranges), rely on the [Repository Overview] block
   at the top of the context — it contains ground-truth statistics and MUST be treated as authoritative.
"""

def build_user_prompt(git_context: str, question: str, repo_stats_block: str = "") -> str:
    """
    Constructs the final rigorous prompt block injecting the context and the question.
    If repo_stats_block is provided, it is prepended before the commit context to give
    the LLM authoritative ground truth for aggregate queries.
    """
    context_parts = []
    if repo_stats_block:
        context_parts.append(repo_stats_block)
    if git_context:
        context_parts.append(git_context)
    full_context = "\n\n".join(context_parts)

    return f"""
<Git History Context>
{full_context}
</Git History Context>

<Developer Question>
{question}
</Developer Question>

Analyze the context above and answer the question as an expert engineer:
"""
