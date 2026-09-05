# Provider authentication status

- Qwen `qwen3.7-plus`: `HTTP 401 authentication_failed`.
- DeepSeek `deepseek-v4-flash`: `HTTP 401 authentication_failed`.
- The repository-external two-line key file mapping is intact: line 1 maps to Qwen and line 2 maps to DeepSeek. After the user's update, both values remained format-valid but their providers again returned HTTP 401.
- No API key value, Authorization header, masked key tail, reasoning content, or formal task evidence was retained.
- No execution lock was created, and no formal candidate request was sent.

## Required external action

Rotate or replace both provider keys in the repository-external secret source, then notify the Agent. The resumed workflow begins by re-running the two schema-only smoke calls before it creates any candidate graph request.
