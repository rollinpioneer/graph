# Provider validation

Frozen configuration was used for all 24 formal requests:

- model: `qwen3.8-max-0902`
- region: `cn-beijing`
- endpoint: `https://dashscope.aliyuncs.com/api/v1`
- SDK: dashscope `1.27.6`
- temperature 0, max_tokens 2048, JSON object, thinking/search/tools off

All 24 requests returned HTTP 200 with a request ID on the first attempt. No authorization, endpoint, or model rejection occurred. The SDK response object did not include an echoed model name; requested model remains the frozen snapshot. Credential value was not logged, hashed, or copied into the repository.
