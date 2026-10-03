# Memory lab runtime

Completed baseline thread memory, advanced persistent `User.md`, bounded compact memory, two benchmark suites, and tests.

Default live model: Ollama Cloud `gpt-oss:120b`. Native HTTP client needs no SDK. Other providers load optional LangChain adapters only when selected. No silent offline fallback.

Run from repository root:

```powershell
python -m pytest src\test_agents.py -q
python src\benchmark.py
# Requires OLLAMA_API_KEY configured locally; sends synthetic dataset to provider.
python src\benchmark.py --live
```

Benchmark defaults to offline and uses disposable isolated memory. Direct agent instances persist profiles in configured state directory. See root [README](../README.md) for configuration, verified offline results, and limits.
