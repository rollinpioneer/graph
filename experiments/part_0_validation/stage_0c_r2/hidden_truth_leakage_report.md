# Hidden truth leakage

Each scene stores simulator pose and withheld dynamic facts only in `hidden_truth_qa.json`. Every QA file records `hidden_truth_used_for_prompt=false`; the formal initial-fact summaries use conservative static/visual records and UNKNOWN for facts not reliably observable. No hidden plan, reward, future state, optimal action, or complete plan is included in the VLM payload.
