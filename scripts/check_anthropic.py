import os
import sys

from crewai import LLM

def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"ENV VAR {name} not defined.")
    return value

model = require_env("ORCHESTRATOR_MODEL")
base_url = require_env("ORCHESTRATOR_BASE_URL")
api_key = require_env("ORCHESTRATOR_API_KEY")

print(f"model={model}  base_url={base_url}  api_key=...{api_key[-4:]}")

llm = LLM(model=model, base_url=base_url, api_key=api_key)
print(llm.call("Hey. Just say hello"))
