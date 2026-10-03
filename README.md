# Phase 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

```
.
├── README.md        # giới thiệu track (file này)
├── Guide.md         # hướng dẫn từng bước
├── Rubric.md        # tiêu chí chấm điểm
├── data/            # dữ liệu benchmark dùng chung
│   ├── conversations.json
│   └── advanced_long_context.json
└── src/             # runtime hoàn thiện: offline + live Ollama Cloud
    ├── model_provider.py
    ├── config.py
    ├── memory_store.py
    ├── agent_baseline.py
    ├── agent_advanced.py
    ├── benchmark.py
    └── test_agents.py
```

Khi chạy, agent sẽ ghi trạng thái (ví dụ `state/profiles/<user>/User.md`) vào thư mục `state/`. Thư mục này đã nằm trong `.gitignore`.

### Vai trò từng file trong `src/`

Các file được liệt kê theo thứ tự nên triển khai:

| File | Vai trò | Thành phần chính |
|---|---|---|
| `model_provider.py` | Khởi tạo chat model cho từng provider | `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `config.py` | Cấu hình chung của lab | `LabConfig` (đường dẫn, ngưỡng compact, model chính + judge), `load_config()` |
| `memory_store.py` | Lõi memory layer | `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread | `BaselineAgent.reply()`, `token_usage()`, `prompt_token_usage()` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact | `AdvancedAgent.reply()`, `_reply_offline()`, `_estimate_prompt_context_tokens()`, `_offline_response()` |
| `benchmark.py` | So sánh hai agent trên hai bộ dữ liệu | `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()` |
| `test_agents.py` | Kiểm chứng hành vi memory | test `User.md`, compact trigger, cross-session recall, giảm prompt load |

### Luồng xử lý một lượt của Advanced Agent

```
message người dùng
  → extract_profile_updates()      # trích fact ổn định: tên, nơi ở, nghề, style...
  → ghi vào User.md                # persistent memory
  → CompactMemoryManager.append()  # short-term memory, tự compact khi vượt ngưỡng
  → prompt = User.md + summary + recent messages
  → sinh câu trả lời → cập nhật bộ đếm token
```

Baseline Agent chỉ giữ danh sách message theo `thread_id`. Sang thread mới, nó **phải quên** toàn bộ fact cũ.

Cả hai agent nên có **chế độ offline** cho ra kết quả lặp lại được, để benchmark và test chạy được mà không cần API key. Chế độ live (LangChain/LangGraph) là phần mở rộng.

## Dữ liệu benchmark

| File | Nội dung | Mục tiêu |
|---|---|---|
| `data/conversations.json` | 10 hội thoại khoảng 10 lượt, user `dungct`, kèm `recall_questions` | Standard benchmark: đo recall qua nhiều phiên bình thường |
| `data/advanced_long_context.json` | 1 hội thoại 16 lượt rất dài, user `dungct_stress` | Long-context stress benchmark: ép compact xảy ra nhiều lần |

Mỗi hội thoại có dạng:

```json
{
  "id": "conv-01",
  "user_id": "dungct",
  "turns": ["...", "..."],
  "recall_questions": [
    { "question": "...", "expected_contains": ["DũngCT", "cà phê sữa đá"] }
  ]
}
```

`recall_questions` được hỏi ở **thread mới**. Điểm recall dựa trên số chuỗi trong `expected_contains` xuất hiện trong câu trả lời.

Dữ liệu cố tình chứa các tình huống khó:

- **correction**: nơi ở đổi giữa Đà Nẵng và Huế, agent phải giữ fact mới nhất
- **nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa
- **ngữ cảnh dài**: nhiều đoạn tin tức dài trong stress test để làm lộ chi phí prompt của baseline

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Setup môi trường

Python `>= 3.11`. Offline và Ollama Cloud dùng thư viện chuẩn; test dùng `pytest`. Chạy PowerShell từ root repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest src\test_agents.py -q
.\.venv\Scripts\python.exe src\benchmark.py
```

Live mặc định: Ollama Cloud, `gpt-oss:120b`, endpoint `https://ollama.com/api/chat`.

```powershell
$env:LLM_PROVIDER = 'ollama'
$env:LLM_MODEL = 'gpt-oss:120b'
$env:OLLAMA_BASE_URL = 'https://ollama.com'
# Set OLLAMA_API_KEY locally through your secret manager or environment settings.
.\.venv\Scripts\python.exe src\benchmark.py --live
```

`--live` gửi dữ liệu benchmark giả lập tới provider và có thể dùng quota. Không gửi code, credentials hoặc dữ liệu riêng tư. Không dán API key vào chat hay commit. Thiếu key/lỗi mạng sẽ báo lỗi, không âm thầm chuyển sang offline. Ollama local: đặt `OLLAMA_BASE_URL=http://localhost:11434` và model đã tải.

Nếu dùng `.env` (đã ignore), cài `python-dotenv`; biến môi trường hiện tại có ưu tiên cao hơn. Provider khác chỉ cần cài package tương ứng: `langchain-openai` (openai/custom/openrouter), `langchain-google-genai` (gemini), `langchain-anthropic` (anthropic). Biến: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `<PROVIDER>_API_KEY`, `<PROVIDER>_BASE_URL`, `COMPACT_THRESHOLD_TOKENS` (1800), `COMPACT_KEEP_MESSAGES` (6). `CUSTOM_BASE_URL` bắt buộc cho custom.

Agent API: `reply(user_id, thread_id, message)` trả `answer`, `agent_tokens`, `prompt_tokens`. Khởi tạo với `force_offline=True` để không gọi mạng. Advanced lưu profile qua lần khởi động mới; thread/summary chỉ nằm trong RAM. User ID chỉ nhận chữ ASCII, số, `_`, `-`; thread không được dùng chung giữa user.

## Chạy benchmark và test

Sau khi hoàn thiện `src/`, chạy từ root repo:

```bash
python src/benchmark.py
```

```bash
pytest src/test_agents.py -v
```

Benchmark in **Standard Benchmark** và **Long-Context Stress Benchmark**, đủ 6 chỉ số. Mỗi suite dùng memory tạm riêng, tự xóa sau khi chạy; không thay profile đang có. Recall hỏi trong thread mới sau từng conversation, trước correction của conversation kế tiếp. Bộ đếm bao gồm cả lượt recall.

Kết quả offline đã kiểm chứng (ngưỡng 1800, giữ 6 messages):

| Suite | Baseline prompt tokens | Advanced prompt tokens | Recall Baseline / Advanced | Advanced compactions |
|---|---:|---:|---|---:|
| Standard | 16612 | 27247 | 0% / 100% | 0 |
| Long-context | 22765 | 18031 | 0% / 100% | 1 |

Advanced tốn thêm context ở chuỗi ngắn vì profile; stress giảm khoảng 20.8% prompt tokens nhờ compact. Token offline ước lượng theo ký tự, không phải billing. Live dùng usage provider khi có, fallback ước lượng nếu không có. Response quality chỉ là proxy lexical, không phải judge độc lập; kết quả offline không chứng minh chất lượng GPT live.

Fact extraction bảo thủ theo pattern tiếng Việt, bỏ câu hỏi/câu đùa/giả định và cập nhật field mới thay field cũ. Profile ghi atomic để tránh file dở dang. Chưa có memory decay hoặc hỗ trợ ghi đồng thời nhiều process; dùng một writer cho mỗi user. Summary extractive bị giới hạn kích thước và có thể mất chi tiết; profile giữ facts ổn định, nhưng task dài cần LLM summary có kiểm chứng.

## Cách dùng repo này

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.
