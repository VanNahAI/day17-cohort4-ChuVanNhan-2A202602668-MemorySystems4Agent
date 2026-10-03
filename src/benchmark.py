from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of conversations")
    ids = set()
    for conversation in data:
        if not isinstance(conversation, dict):
            raise ValueError("Conversation must be an object")
        if not all(isinstance(conversation.get(key), str) and conversation[key] for key in ("id", "user_id")):
            raise ValueError("Conversation requires nonempty id and user_id")
        if conversation["id"] in ids:
            raise ValueError("Conversation ids must be unique")
        ids.add(conversation["id"])
        turns = conversation.get("turns")
        questions = conversation.get("recall_questions")
        if not isinstance(turns, list) or not all(isinstance(turn, str) for turn in turns):
            raise ValueError("turns must be a list of strings")
        if not isinstance(questions, list):
            raise ValueError("recall_questions must be a list")
        for question in questions:
            if not isinstance(question, dict) or not isinstance(question.get("question"), str):
                raise ValueError("Recall question must contain question text")
            expected = question.get("expected_contains")
            if not isinstance(expected, list) or not expected or not all(isinstance(fact, str) and fact for fact in expected):
                raise ValueError("expected_contains must contain nonempty strings")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    normalize = lambda text: unicodedata.normalize("NFC", text).casefold()
    hits = sum(normalize(fact) in normalize(answer) for fact in expected)
    return 1.0 if expected and hits == len(expected) else 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    # ponytail: lexical proxy, not a model judge; add blinded human/LLM evaluation for real quality claims.
    return recall_points(answer, expected) * (1.0 if len(answer) <= 1000 else 0.8)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    users = {conversation["user_id"] for conversation in conversations}
    size = lambda user: agent.memory_file_size(user) if hasattr(agent, "memory_file_size") else 0
    before = sum(size(user) for user in users)
    threads = []
    recall = []
    quality = []
    for conversation in conversations:
        user = conversation["user_id"]
        thread = f"chat:{conversation['id']}"
        threads.append(thread)
        for turn in conversation["turns"]:
            agent.reply(user, thread, turn)
        for index, question in enumerate(conversation["recall_questions"]):
            fresh = f"recall:{conversation['id']}:{index}"
            threads.append(fresh)
            answer = agent.reply(user, fresh, question["question"])["answer"]
            recall.append(recall_points(answer, question["expected_contains"]))
            quality.append(heuristic_quality(answer, question["expected_contains"]))
    return BenchmarkRow(agent_name, sum(agent.token_usage(t) for t in threads),
                        sum(agent.prompt_token_usage(t) for t in threads),
                        sum(recall) / len(recall) if recall else 0,
                        sum(quality) / len(quality) if quality else 0,
                        sum(size(user) for user in users) - before,
                        sum(agent.compaction_count(t) for t in threads))


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
               "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
                  f"{row.recall_score:.1%}", f"{row.response_quality:.1%}", str(row.memory_growth_bytes), str(row.compactions)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory benchmark; offline by default")
    parser.add_argument("--live", action="store_true", help="Send synthetic benchmark turns to configured LLM")
    args = parser.parse_args()
    config = load_config()
    print(f"Mode: {'live' if args.live else 'offline'}; model: {config.model.provider}/{config.model.model_name}")
    print("Response quality: lexical heuristic. Offline tokens: character estimates; live tokens: provider usage when available.")
    for heading, filename in [("Standard Benchmark", "conversations.json"),
                              ("Long-Context Stress Benchmark", "advanced_long_context.json")]:
        conversations = load_conversations(config.data_dir / filename)
        with TemporaryDirectory(prefix="memory-lab-") as directory:
            isolated = replace(config, state_dir=Path(directory))
            rows = [run_agent_benchmark("Baseline", BaselineAgent(isolated, force_offline=not args.live), conversations, isolated),
                    run_agent_benchmark("Advanced", AdvancedAgent(isolated, force_offline=not args.live), conversations, isolated)]
            print(f"\n## {heading}\n{format_rows(rows)}")


if __name__ == "__main__":
    main()