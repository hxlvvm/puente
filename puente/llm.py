"""LLM backends: Gemini over REST (key from GEMINI_API_KEY) and a local Hugging Face chat model on a GPU."""
import json
import os
import time
import urllib.error
import urllib.request

GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class Gemini:
    def __init__(self, model="gemini-3.5-flash-lite", min_interval=4.0):
        self.model, self.min_interval, self._last = model, min_interval, 0.0
        self.key = os.environ["GEMINI_API_KEY"]
        self.name = model

    def generate(self, prompts, json_mode=True):
        return [self._one(p, json_mode) for p in prompts]

    def _one(self, prompt, json_mode):
        cfg = {"temperature": 0, "maxOutputTokens": 4096}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}).encode()
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(GEMINI.format(model=self.model), data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.load(r)
                return "".join(p.get("text", "") for p in resp["candidates"][0]["content"].get("parts", []))
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503):
                    time.sleep(min(60, 10 * (attempt + 1)))
                    continue
                return ""
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError):
                time.sleep(5)
        # quota or outage: stop instead of recording an empty answer as a model failure
        raise RuntimeError("Gemini unavailable after retries (quota or outage); rerun later to resume")


class LocalHF:
    """Greedy batched generation with a chat-template model (e.g. Qwen/Qwen2.5-7B-Instruct)."""

    def __init__(self, model_id, batch=16, max_new_tokens=900):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.name, self.batch, self.max_new = torch, model_id.split("/")[-1], batch, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda").eval()

    def generate(self, prompts, json_mode=True):
        out = []
        for s in range(0, len(prompts), self.batch):
            chats = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                  add_generation_prompt=True) for p in prompts[s:s + self.batch]]
            enc = self.tok(chats, return_tensors="pt", padding=True).to("cuda")
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out
