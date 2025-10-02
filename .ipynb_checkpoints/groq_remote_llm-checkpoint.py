from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from pydantic import Field
import os, requests

class GroqRemoteLLM(LLM):
    api_url: str = Field(default_factory=lambda: os.getenv("GROQ_API_URL"))
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    model: str   = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        return "groq-remote-llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "url": self.api_url}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
          "model": self.model,
          "messages": [
            {"role":"system",
             "content": "You are a helpful assistant. Answer only using the retrieved document content provided in the prompt. "
                        "If the information is not present in the provided content, reply: 'Not stated in the document.'"},
            {"role":"user", "content": prompt},
          ],
          "temperature": 0.0,
          "max_tokens": 256,
        }
        
        
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
        doc = resp.json()
        return doc["choices"][0]["message"]["content"]
