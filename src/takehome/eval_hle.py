"""Humanity's Last Exam: generate with a served model, judge with the official prompt.

    python -m takehome.eval_hle generate --name base --api http://localhost:18000/v1 --out results/base/hle/run1
    python -m takehome.eval_hle judge --out results/base/hle/run1

Protocol (identical for every evaluated model; see configs/eval/hle.yaml):
- cais/hle test split at the revision in configs/eval/hle.lock.json, all 2,500 questions
  including image questions (Qwen3.5-9B is multimodal), no tools.
- Official system prompt and judge prompt from centerforaisafety/hle@22ed3074
  (hle_eval/run_model_predictions.py, run_judge_results.py), judge o3-mini-2025-01-31.
- Thinking on; temperature 1.0, top_p 0.95, top_k 20, max_tokens 32768, seed = crc32(id).
- Accuracy is over all questions: a failed or truncated generation counts as incorrect and is
  listed in failures.json. Text-only accuracy is reported separately and labeled as such.
"""

import argparse
import asyncio
import json
import time
import zlib
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel

SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""

JUDGE_MODEL = "o3-mini-2025-01-31"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768, "extra_body": {"top_k": 20}}


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


def load_hle():
    lock = json.load(open("configs/eval/hle.lock.json"))
    ds = load_dataset(lock["repo"], split=lock["split"], revision=lock["revision"])
    assert len(ds) == lock["n_tasks"], (len(ds), lock["n_tasks"])
    return ds


def messages(q):
    content = [{"type": "text", "text": q["question"]}]
    if q["image"]:
        content.append({"type": "image_url", "image_url": {"url": q["image"]}})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


async def retry(fn, tries=5, conn_wait_s=3 * 3600):
    """Exponential backoff (1 s .. 60 s). Connection failures (the serving endpoint is between
    HPC job windows) are retried for up to conn_wait_s without consuming `tries`."""
    import openai
    delay, i, waited = 1.0, 0, 0.0
    while True:
        try:
            return await fn()
        except (openai.APIConnectionError, openai.APITimeoutError):
            if waited > conn_wait_s:
                raise
        except Exception:  # rate limits, transient server errors
            i += 1
            if i >= tries:
                raise
        await asyncio.sleep(min(60.0, delay))
        waited += min(60.0, delay)
        delay *= 2


async def generate(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done = {json.loads(l)["id"] for l in open(pred_path)} if pred_path.exists() else set()
    ds = load_hle()
    todo = [q for q in ds if q["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    client = AsyncOpenAI(base_url=a.api, api_key="EMPTY", timeout=3600, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    f = open(pred_path, "a")

    async def one(q):
        async with sem:
            t0 = time.time()
            rec = {"id": q["id"], "model": a.name, "has_image": bool(q["image"])}
            try:
                r = await retry(lambda: client.chat.completions.create(
                    model=a.name, messages=messages(q), seed=zlib.crc32(q["id"].encode()), **SAMPLING))
                m = r.choices[0].message
                rec.update(response=m.content, reasoning_chars=len(getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or ""),
                           finish_reason=r.choices[0].finish_reason, usage=r.usage.model_dump() if r.usage else None)
            except Exception as e:
                rec.update(response=None, error=f"{type(e).__name__}: {e}"[:2000])
            rec["latency_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()

    await asyncio.gather(*(one(q) for q in todo))


async def judge(a):
    out = Path(a.out)
    preds = {r["id"]: r for r in map(json.loads, open(out / "predictions.jsonl"))}
    judged_path = out / "judged.jsonl"
    judged = {json.loads(l)["id"] for l in open(judged_path)} if judged_path.exists() else set()
    key = Path(a.key_file).expanduser().read_text().strip()
    client = AsyncOpenAI(api_key=key, timeout=300, max_retries=0)
    ds = load_hle()
    sem = asyncio.Semaphore(a.concurrency)
    f = open(judged_path, "a")

    async def one(q):
        p = preds.get(q["id"])
        if p is None or q["id"] in judged:
            return
        rec = {"id": q["id"], "judge": JUDGE_MODEL}
        if not p.get("response"):
            rec.update(correct="no", judge_error="no response (generation failure)")
        else:
            async with sem:
                try:
                    r = await retry(lambda: client.beta.chat.completions.parse(
                        model=JUDGE_MODEL, max_completion_tokens=4096, response_format=ExtractedAnswer,
                        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                            question=q["question"], correct_answer=q["answer"], response=p["response"])}]))
                    c = r.choices[0].message.parsed
                    rec.update(model_answer=c.extracted_final_answer, reasoning=c.reasoning, correct=c.correct,
                               confidence=c.confidence)
                except Exception as e:
                    rec.update(correct=None, judge_error=f"{type(e).__name__}: {e}"[:1000])
        f.write(json.dumps(rec) + "\n")
        f.flush()

    await asyncio.gather(*(one(q) for q in ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "judge"])
    ap.add_argument("--name")
    ap.add_argument("--api", default="http://localhost:18000/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--key-file", default="~/.config/openai/api_key")
    a = ap.parse_args()
    asyncio.run(generate(a) if a.stage == "generate" else judge(a))


if __name__ == "__main__":
    main()
