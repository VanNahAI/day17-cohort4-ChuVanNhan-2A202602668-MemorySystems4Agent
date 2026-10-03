from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import load_conversations, recall_points, run_agent_benchmark
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import ProviderConfig, build_chat_model, normalize_provider


def make_config(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    model = ProviderConfig("ollama", "gpt-oss:120b", 0)
    return LabConfig(root, root / "data", tmp_path, 300, 2, model, model)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    assert store.file_size("alice") == 0
    assert store.read_text("alice") == "# User\n"
    store.write_text("alice", "# User\n- name: Alice\n")
    assert store.edit_text("alice", "Alice", "An")
    assert not store.edit_text("alice", "missing", "Bob")
    assert "An" in store.read_text("alice")
    store.update_facts("alice", {"location": "Huế"})
    store.update_facts("alice", {"location": "Đà Nẵng"})
    assert "Huế" not in store.read_text("alice")
    assert store.file_size("alice") == len(store.read_text("alice").encode())
    for invalid in ["../alice", "..", "", "a/b", "a\\b"]:
        with pytest.raises(ValueError):
            store.path_for(invalid)


def test_compact_trigger(tmp_path: Path) -> None:
    memory = CompactMemoryManager(100, 2)
    for index in range(8):
        memory.append("thread", "user", f"Turn {index}: " + "context " * 100)
    context = memory.context("thread")
    assert memory.compaction_count("thread") > 0
    assert len(context["messages"]) == 2
    assert 0 < len(context["summary"]) <= 100
    assert "Turn 7" in context["messages"][-1]["content"]
    assert estimate_tokens("   ") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for agent in (advanced, baseline):
        agent.reply("alice", "first", "Mình tên là An. Mình ở Huế.")
        assert "An" in agent.reply("alice", "first", "Mình tên gì?")["answer"]
    restarted = AdvancedAgent(config, force_offline=True)
    assert "An" in restarted.reply("alice", "new", "Mình tên gì?")["answer"]
    assert "An" not in baseline.reply("alice", "new", "Mình tên gì?")["answer"]
    assert "An" not in restarted.reply("bob", "bob-thread", "Mình tên gì?")["answer"]
    with pytest.raises(ValueError):
        restarted.reply("bob", "new", "Xin chào")


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for index in range(20):
        text = f"Tin tức {index}: " + "Ngữ cảnh dài về tin tức. " * 150
        baseline.reply("alice", "long", text)
        advanced.reply("alice", "long", text)
    assert advanced.compaction_count("long") > 1
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.5


def test_corrections_and_noise(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for text in ["Mình ở Huế và đang làm backend engineer.",
                 "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
                 "Giờ mình đang ở Đà Nẵng.",
                 "Mình đùa là chuyển sang product manager, nhưng đó chỉ là câu đùa.",
                 "Nếu nhắc Hà Nội thì đó là nơi đi họp."]:
        agent.reply("alice", "first", text)
    facts = agent.profile_store.facts("alice")
    assert facts["location"] == "Đà Nẵng"
    assert facts["profession"] == "MLOps engineer"
    assert extract_profile_updates("Hiện tại mình đang ở đâu?") == {}


def test_benchmark_datasets(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for filename in ["conversations.json", "advanced_long_context.json"]:
        conversations = load_conversations(config.data_dir / filename)
        advanced = run_agent_benchmark("Advanced", AdvancedAgent(config, True), conversations, config)
        baseline = run_agent_benchmark("Baseline", BaselineAgent(config, True), conversations, config)
        assert advanced.recall_score == 1
        assert baseline.recall_score == 0
        assert advanced.memory_growth_bytes > 0
        assert baseline.memory_growth_bytes == 0
    assert recall_points("a", ["a", "b"]) == 0.5


def test_ollama_cloud_request_and_usage(tmp_path: Path) -> None:
    model = build_chat_model(ProviderConfig("ollama", "gpt-oss:120b", 0, "test-key", "https://ollama.com"))
    response = SimpleNamespace()
    from io import BytesIO
    payload = {"message": {"content": "Xin chào"}, "prompt_eval_count": 50, "eval_count": 7}
    with patch("model_provider.urlopen", return_value=BytesIO(json.dumps(payload).encode())) as request:
        response = model.invoke([{"role": "user", "content": "Xin chào"}])
    call = request.call_args.args[0]
    assert call.full_url == "https://ollama.com/api/chat"
    assert call.get_header("Authorization") == "Bearer test-key"
    assert json.loads(call.data)["model"] == "gpt-oss:120b"
    assert response.usage_metadata == {"input_tokens": 50, "output_tokens": 7}
    for cls in (BaselineAgent, AdvancedAgent):
        with patch(f"{cls.__module__}.build_chat_model", return_value=SimpleNamespace(invoke=lambda _: response)):
            agent = cls(make_config(tmp_path))
            assert agent.reply("alice", "live", "Mình tên là An.")["answer"] == "Xin chào"
            assert agent.token_usage("live") == 7
            assert agent.prompt_token_usage("live") == 50
    assert normalize_provider("anthorpic") == "anthropic"
    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        build_chat_model(ProviderConfig("ollama", "gpt-oss:120b", 0))


def test_live_failure_does_not_commit_turn(tmp_path: Path) -> None:
    def fail(_):
        raise TimeoutError("test timeout")
    for cls in (BaselineAgent, AdvancedAgent):
        with patch(f"{cls.__module__}.build_chat_model", return_value=SimpleNamespace(invoke=fail)):
            agent = cls(make_config(tmp_path))
            with pytest.raises(TimeoutError):
                agent.reply("alice", "failed", "Mình tên là An.")
            assert agent.token_usage("failed") == 0
            if cls is AdvancedAgent:
                assert agent.compact_memory.context("failed")["messages"] == []
                assert agent.memory_file_size("alice") == 0
            else:
                assert agent.sessions["failed"].messages == []