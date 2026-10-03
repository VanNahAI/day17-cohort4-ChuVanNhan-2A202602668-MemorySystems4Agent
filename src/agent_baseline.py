from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


SYSTEM_PROMPT = "Trả lời bằng tiếng Việt, ngắn gọn. Chỉ dùng thông tin người dùng đã cung cấp; không đoán facts còn thiếu."


def offline_answer(facts: dict[str, str], message: str) -> str:
    if re.search(r"\?|nhắc|tóm tắt|thử nhớ", message, re.I):
        return "; ".join(facts.values()) if facts else "Mình chưa có thông tin về bạn trong phiên này."
    return "Đã ghi nhận thông tin trong cuộc trò chuyện."


def response_counts(response, prompt: list[dict[str, str]]) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    return (usage.get("output_tokens") if usage.get("output_tokens") is not None else estimate_tokens(response.content),
            usage.get("input_tokens") if usage.get("input_tokens") is not None else sum(estimate_tokens(m["content"]) for m in prompt))


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Thread-only memory; new threads start without personal facts."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id in self.thread_users and self.thread_users[thread_id] != user_id:
            raise ValueError("Thread belongs to another user")
        self.thread_users[thread_id] = user_id
        if self.force_offline:
            return self._reply_offline(thread_id, message)
        state = self.sessions.setdefault(thread_id, SessionState())
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + state.messages + [{"role": "user", "content": message}]
        response = self.langchain_agent.invoke(prompt)
        output, processed = response_counts(response, prompt)
        state.messages.extend([prompt[-1], {"role": "assistant", "content": response.content}])
        state.token_usage += output
        state.prompt_tokens_processed += processed
        return {"answer": response.content, "agent_tokens": output, "prompt_tokens": processed}

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        facts = {}
        for turn in state.messages:
            if turn["role"] == "user":
                facts.update(extract_profile_updates(turn["content"]))
        answer = offline_answer(facts, message)
        processed = estimate_tokens(SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in state.messages)
        output = estimate_tokens(answer)
        state.messages.append({"role": "assistant", "content": answer})
        state.token_usage += output
        state.prompt_tokens_processed += processed
        return {"answer": answer, "agent_tokens": output, "prompt_tokens": processed}

    def _maybe_build_langchain_agent(self):
        return build_chat_model(self.config.model)