"""
gitpry/llm/chat_session.py

Manages the state for a single `git pry chat` session:
- Conversation history (question + context + answer triples)
- Branch and retrieval mode settings
- Cross-turn combined query construction for smarter RAG retrieval
- messages[] array building for Ollama /api/chat
"""
from dataclasses import dataclass, field
from typing import Optional
from gitpry.llm.prompts import SYSTEM_PROMPT


@dataclass
class ChatTurn:
    question: str
    context: str   # The git context injected for this turn
    answer: str


@dataclass
class ChatSession:
    branch: Optional[str] = None
    no_rag: bool = False
    repo_stats_block: str = ""    # Cached after first turn
    max_turns: int = 10
    history: list = field(default_factory=list)   # list[ChatTurn]

    def add_turn(self, question: str, context: str, answer: str):
        """Record a completed turn. Trim history if over the limit."""
        self.history.append(ChatTurn(question=question, context=context, answer=answer))
        if len(self.history) > self.max_turns:
            self.history.pop(0)  # Drop oldest turn

    def build_combined_query(self, current_question: str) -> str:
        """
        Build a retrieval query that merges the last 3 questions with the
        current question. This helps RAG find relevant context for
        follow-up questions like "tell me more about that commit".
        """
        recent_questions = [t.question for t in self.history[-3:]]
        all_questions = recent_questions + [current_question]
        return " ".join(all_questions)

    def build_messages(self, current_context: str, current_question: str) -> list:
        """
        Build the Ollama /api/chat messages array:
        [system] + [(user+ctx, assistant)] * history + [user+ctx (current)]
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject historical turns
        for turn in self.history:
            user_content = _format_user_message(
                turn.context, turn.question, self.repo_stats_block, is_first=(turn == self.history[0])
            )
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": turn.answer})

        # Current turn (always inject stats block on first turn; omit after to save tokens)
        is_first_turn = len(self.history) == 0
        current_user_content = _format_user_message(
            current_context, current_question, self.repo_stats_block, is_first=is_first_turn
        )
        messages.append({"role": "user", "content": current_user_content})

        return messages

    def clear(self):
        """Reset conversation history, keeping branch and mode."""
        self.history.clear()

    @property
    def turn_count(self) -> int:
        return len(self.history)


def _format_user_message(context: str, question: str, stats_block: str, is_first: bool) -> str:
    """
    Format a single user turn message for /api/chat.
    Only injects the repo stats block on the first turn to save tokens.
    """
    parts = []
    if is_first and stats_block:
        parts.append(stats_block)
    if context:
        parts.append(f"<Git History Context>\n{context}\n</Git History Context>")
    parts.append(f"<Developer Question>\n{question}\n</Developer Question>")
    parts.append("Analyze the context above and answer the question as an expert engineer:")
    return "\n\n".join(parts)
