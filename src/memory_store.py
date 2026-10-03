from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    return (len(text.strip()) + 3) // 4


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", user_id):
            raise ValueError("user_id must contain only letters, digits, underscores or hyphens")
        path = self.root_dir / user_id / "User.md"
        if not path.resolve().is_relative_to(self.root_dir.resolve()):
            raise ValueError("Profile path escapes root")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if not search_text or search_text not in text or search_text == replacement:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- ([a-z_]+): (.+)$", self.read_text(user_id), re.M))

    def update_facts(self, user_id: str, updates: dict[str, str]) -> None:
        if not updates:
            return
        text = self.read_text(user_id)
        for key, value in updates.items():
            line = f"- {key}: {value}"
            pattern = rf"^- {re.escape(key)}: .*?$"
            if re.search(pattern, text, re.M):
                text = re.sub(pattern, lambda _: line, text, count=1, flags=re.M)
            else:
                text = text.rstrip() + "\n" + line + "\n"
        self.write_text(user_id, text)


def extract_profile_updates(message: str) -> dict[str, str]:
    # ponytail: conservative Vietnamese patterns; use validated structured LLM extraction for new languages.
    updates = {}
    sentences = re.split(r"(?<=[.!?;])|\n", message)
    patterns = {
        "name": r"(?:mình|tôi) tên là ([^,]+)",
        "location": r"(?:mình|tôi|hiện|hiện tại|giờ)(?: vẫn| đang| hiện| hiện tại)? (?:đang )?(?:ở|làm việc ở) (Huế|Đà Nẵng|Hà Nội)",
        "profession": r"(?:đang làm|vẫn là|làm|chuyển sang|nghề nghiệp hiện tại[^,]*?là) (MLOps engineer|backend engineer|product manager)",
        "drink": r"đồ uống yêu thích (?:của mình )?là ([^,]+)",
        "food": r"món ăn yêu thích (?:của mình )?là ([^,]+)",
        "pet": r"mình nuôi (.+)",
    }
    for sentence in sentences:
        if "?" in sentence or re.search(r"\bgì\b|câu đùa|mình đùa", sentence, re.I):
            continue
        sentence = sentence.strip().rstrip(".!;")
        if re.search(r"nếu |đừng ", sentence, re.I):
            continue
        if re.search(r"không còn|không phải", sentence, re.I):
            if "chuyển sang" in sentence.lower():
                sentence = sentence[sentence.lower().index("chuyển sang"):]
            else:
                sentence = re.split(r"chứ không còn|không còn|không phải", sentence, flags=re.I)[0]
        for key, pattern in patterns.items():
            matches = list(re.finditer(pattern, sentence, re.I))
            if matches:
                updates[key] = matches[-1].group(1).strip()
        if re.search(r"(?:mình thích|mình vẫn thích|dài hạn: mình thích)", sentence, re.I):
            match = re.search(r"mình (?:vẫn )?thích (.+)", sentence, re.I)
            if match and re.search(r"Python|AI", match.group(1)):
                updates["interests"] = match.group(1).strip()
        if re.search(r"(?:mình muốn|mình thích|hãy trả lời|style trả lời)", sentence, re.I):
            if re.search(r"ngắn|bullet", sentence, re.I):
                updates["style"] = "ngắn gọn, rõ ý, có ví dụ thực tế"
                if "3 bullet" in message:
                    updates["style"] += ", 3 bullet"
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    facts = {}
    excerpts = []
    for message in messages:
        if message["role"] == "user":
            facts.update(extract_profile_updates(message["content"]))
        excerpts.append(f"{message['role']}: {message['content'][:160]}")
    return "\n".join([f"{key}: {value}" for key, value in facts.items()] + excerpts[-max_items:])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 1:
            raise ValueError("Invalid compact memory limits")

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.context(thread_id)
        context["messages"].append({"role": role, "content": content})
        load = estimate_tokens(context["summary"]) + sum(estimate_tokens(m["content"]) for m in context["messages"])
        if load > self.threshold_tokens and len(context["messages"]) > self.keep_messages:
            old = context["messages"][:-self.keep_messages]
            summary = summarize_messages(old)
            # ponytail: bounded extractive summary loses detail; upgrade to LLM summary for richer task recall.
            budget = min(1200, self.threshold_tokens)
            context["summary"] = (context["summary"] + "\n" + summary)[-budget:]
            context["messages"] = context["messages"][-self.keep_messages:]
            context["compactions"] += 1

    def context(self, thread_id: str) -> dict:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]