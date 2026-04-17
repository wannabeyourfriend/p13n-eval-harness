"""PrefEval — unified runner: generation + classification + LLM judge + accuracy.

Upstream: https://github.com/amazon-science/PrefEval

Stages
------
- gen       : generate responses to preference-related questions (parallel)
- cls       : MCQ classification — does the model pick the preference-aligned option
- judge     : LLM-as-judge over generation outputs on 4 dims (acknow/violate/
              hallucinate/helpful). Ported from upstream boto3+Claude to
              OpenAI-compatible — so the same endpoint that serves the target
              model (or a separate judge endpoint) can grade.
- accuracy  : aggregate judge outputs into the paper's Preference Adherence Accuracy.
- all       : gen → cls → judge → accuracy

CLI is unified with other benchmarks: --api-base / --model / --workers /
--output-dir. Extra flags: --topic, --inter-turns, --task, --stage.

Parallelism: every stage uses multibench.client.LLMClient.chat_batch, so a
single --workers flag controls concurrency across gen/cls/judge.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

from ...args import add_common_llm_args
from ...client import LLMClient, client_from_args
from ...utils import atomic_write_json, benchmark_data_dir, ensure_dir

from .utils.common_utils import (
    ALL_TOPICS, COT_PROMPT, REMINDER, extract_multi_turn_conversation,
)
from .utils.explicit_utils import (
    create_user_pref_message, get_question_prompt,
)


JUDGE_PROMPT_DIR = Path(__file__).parent / "error_type"
JUDGE_FILES = {
    "acknow":      "check_acknowledge.txt",
    "violate":     "check_violation.txt",
    "hallucinate": "check_hallucination.txt",
    "helpful":     "check_helpful.txt",
}


# ---------------- helpers ----------------

def _data_root() -> Path:
    return benchmark_data_dir("prefeval")


def _load_topic_data(pref_form: str, topic: str) -> list[dict]:
    """Load the raw benchmark dataset for a given (pref_form, topic)."""
    root = _data_root()
    if pref_form == "explicit":
        path = root / "explicit_preference" / f"{topic}.json"
    else:
        path = root / "implicit_preference" / "choice-based" / f"{topic}.json"
    with path.open() as f:
        return json.load(f)


def _load_turns_data() -> list[dict]:
    path = _data_root() / "filtered_inter_turns.json"
    with path.open() as f:
        return json.load(f)


def _build_inter_messages(inter_turns: int) -> list[dict]:
    if inter_turns <= 0:
        return []
    turns = _load_turns_data()
    flat = []
    for t in turns:
        flat.extend(t["conversation"])
    return extract_multi_turn_conversation(flat, inter_turns, model_type="claude")


def _paths(args, stage: str) -> Path:
    """Return the save_file for a given stage.

    All results land under args.output_dir/{stage}/{pref_form}/{task}/{topic}/
    keyed by model + topic + inter_turns.
    """
    subdir = Path(args.output_dir) / stage / args.pref_form / args.task / args.topic
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{args.model.replace('/', '_')}_{args.topic}_{args.inter_turns}interturn.json"


# ---------------- stage: generation ----------------

def stage_gen(client: LLMClient, args) -> Path:
    if args.pref_form != "explicit":
        raise NotImplementedError("Only --pref-form=explicit is currently migrated. "
                                  "Implicit form (persona/choice conversations) pending.")

    topic_data = _load_topic_data(args.pref_form, args.topic)
    if args.max_items:
        topic_data = topic_data[: args.max_items]

    save_file = _paths(args, "generation_results")
    if save_file.exists():
        with save_file.open() as f:
            existing = json.load(f)
        if len(existing) >= len(topic_data) and all("response_to_q" in t for t in existing[:len(topic_data)]):
            print(f"[gen] already complete at {save_file}")
            return save_file
        # Merge: use existing where present
        if len(existing) >= len(topic_data):
            topic_data = existing[:len(topic_data)]

    inter_messages = _build_inter_messages(args.inter_turns)
    system_prompt = "You are a helpful assistant."

    # Step 1: pref_generation — ask model to acknowledge the preference
    pref_items = [
        create_user_pref_message(t["preference"], "claude", system_prompt)
        for t in topic_data
    ]
    print(f"[gen] Step 1: eliciting pref_generation for {len(pref_items)} items...")
    pref_responses = client.chat_batch(
        pref_items, build_messages=lambda m: m,
        workers=args.workers, max_tokens=args.max_tokens,
        temperature=args.temperature, system=system_prompt,
        desc=f"PrefEval gen·pref {args.topic}",
    )
    pref_responses = ["" if isinstance(r, Exception) else r for r in pref_responses]

    # Step 2: build final question messages and generate answers
    messages_list = []
    for t, pref_gen in zip(topic_data, pref_responses):
        msgs = get_question_prompt(
            preference=t["preference"],
            pref_generation=pref_gen,
            question=t["question"],
            multi_inter_message=inter_messages,
            model_type="claude",
            turn_number=args.inter_turns,
            remind=(args.task == "remind"),
            cot=(args.task == "cot"),
            args=args, max_tokens=args.max_tokens,
            system_prompt=system_prompt,
        )
        messages_list.append(msgs)

    print(f"[gen] Step 2: answering question for {len(messages_list)} items...")
    answers = client.chat_batch(
        messages_list, build_messages=lambda m: m,
        workers=args.workers, max_tokens=args.max_tokens,
        temperature=args.temperature, system=system_prompt,
        desc=f"PrefEval gen·ans {args.topic}",
    )
    answers = ["" if isinstance(r, Exception) else r for r in answers]

    for t, pref_gen, ans in zip(topic_data, pref_responses, answers):
        t["response_to_pref"] = pref_gen
        t["response_to_q"] = ans

    atomic_write_json(save_file, topic_data)
    print(f"[gen] saved {len(topic_data)} responses → {save_file}")
    return save_file


# ---------------- stage: classification (MCQ) ----------------

_MCQ_CHOICE_RE = re.compile(r"<choice>\s*([A-D])", re.IGNORECASE)


def _extract_choice(text: str) -> str | None:
    m = _MCQ_CHOICE_RE.search(text)
    if m:
        return m.group(1).upper()
    # fallback: first standalone A/B/C/D
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else None


def stage_cls(client: LLMClient, args) -> Path:
    """Classification: present 4 options and see if the model picks the
    preference-aligned one. Migrates the logic in upstream
    `benchmark_classification.py` but runs every item in parallel.
    """
    if args.pref_form != "explicit":
        raise NotImplementedError("Only --pref-form=explicit currently migrated for cls.")

    topic_data = _load_topic_data(args.pref_form, args.topic)
    if args.max_items:
        topic_data = topic_data[: args.max_items]

    save_file = _paths(args, "mcq_results")
    inter_messages = _build_inter_messages(args.inter_turns)
    system_prompt = "You are a helpful assistant."

    # Each PrefEval explicit item has "preference", "question", and 4-way MCQ options.
    # We replicate upstream logic: 4 options listed A-D, the correct one being the
    # preference-aligned option (index 0 after shuffle is stored in "aligned_op").
    import random
    rng = random.Random(args.seed)

    items = []
    for t in topic_data:
        if "options" in t:
            options = list(t["options"])
        elif "mcq_options" in t:
            options = list(t["mcq_options"])
        else:
            # No MCQ options present — skip classification for this item
            items.append(None)
            continue
        correct_answer = t.get("aligned_op") or options[0]
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_letter = "ABCD"[shuffled.index(correct_answer)] if correct_answer in shuffled else None

        opts_str = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(shuffled))
        question_text = (
            f"{t['question']}\n\nChoose one option:\n{opts_str}\n\n"
            f"Answer with the letter inside <choice></choice> tags. "
            f"Example: <choice>A</choice>."
        )

        # Build message list: preference + inter filler + question
        msgs = [
            {"role": "user", "content": t["preference"]},
            {"role": "assistant", "content": "Noted."},
        ]
        if inter_messages:
            msgs.extend(inter_messages)
        msgs.append({"role": "user", "content": question_text})
        items.append({
            "messages": msgs,
            "shuffled": shuffled,
            "correct_letter": correct_letter,
        })

    valid_idx = [i for i, it in enumerate(items) if it is not None]
    print(f"[cls] running {len(valid_idx)} MCQ items (skipped {len(items) - len(valid_idx)} without options)")

    responses: list = [None] * len(items)
    if valid_idx:
        valid_items = [items[i] for i in valid_idx]
        out = client.chat_batch(
            valid_items, build_messages=lambda it: it["messages"],
            workers=args.workers, max_tokens=16,
            temperature=args.temperature, system=system_prompt,
            desc=f"PrefEval cls {args.topic}",
        )
        for i, r in zip(valid_idx, out):
            responses[i] = r

    correct = 0
    total = 0
    for t, it, resp in zip(topic_data, items, responses):
        if it is None:
            continue
        total += 1
        text = "" if isinstance(resp, Exception) else (resp or "")
        chosen = _extract_choice(text)
        is_correct = (chosen is not None and chosen == it["correct_letter"])
        correct += int(is_correct)
        t["mcq_response"] = text
        t["mcq_choice"] = chosen
        t["mcq_correct_letter"] = it["correct_letter"]
        t["mcq_correct"] = is_correct
        t["mcq_shuffled_options"] = it["shuffled"]

    acc = correct / max(total, 1)
    print(f"[cls] accuracy: {acc:.2%} ({correct}/{total})")
    atomic_write_json(save_file, topic_data)
    summary = save_file.with_name(save_file.stem + "_summary.json")
    atomic_write_json(summary, {
        "model": args.model, "topic": args.topic, "inter_turns": args.inter_turns,
        "accuracy": acc, "correct": correct, "total": total,
    })
    return save_file


# ---------------- stage: LLM judge ----------------

def _load_judge_prompts() -> dict[str, str]:
    out = {}
    for k, fname in JUDGE_FILES.items():
        out[k] = (JUDGE_PROMPT_DIR / fname).read_text(encoding="utf-8")
    return out


def _parse_explanation_answer(text: str) -> tuple[str, str]:
    soup = BeautifulSoup(text, "html.parser")
    expl = soup.find("explanation")
    ans = soup.find("answer")
    return (expl.text.strip() if expl else "",
            ans.text.strip() if ans else "")


def _parse_preference_answer(text: str) -> tuple[str, str]:
    soup = BeautifulSoup(text, "html.parser")
    pref = soup.find("preference")
    ans = soup.find("answer")
    return (pref.text.strip() if pref else "",
            ans.text.strip() if ans else "")


def stage_judge(client: LLMClient, args, *, judge_client: LLMClient | None = None) -> Path:
    """Port of upstream llm_based_evaluation_errortypes.py to OpenAI-compatible.

    Loads the generation output, runs 4 judge prompts per item:
      1. acknow:     did the assistant acknowledge the preference?
      2. violate:    does the response violate the preference?
      3. hallucinate: did the assistant mis-quote / hallucinate the preference?
      4. helpful:    is the response helpful?
    Each answer is parsed for <answer>...</answer>. Results stored in
    `evaluation_error_analysis` on each task.
    """
    gen_file = _paths(args, "generation_results")
    if not gen_file.exists():
        raise FileNotFoundError(f"No generation results at {gen_file} — run gen first.")
    with gen_file.open() as f:
        gen_data = json.load(f)
    if args.max_items:
        gen_data = gen_data[: args.max_items]

    judge_prompts = _load_judge_prompts()
    judge = judge_client or client
    system_prompt = ("You are a helpful assistant in evaluating an AI assistant's response. "
                     "You should be fair and strict and follow the user's instruction")
    max_tokens = 300

    save_file = _paths(args, "judged_results")

    # Flatten: every (item, metric) pair where the metric hasn't been evaluated yet.
    pending: list[tuple[int, str, str]] = []  # (task_idx, metric, prompt_text)
    for idx, task in enumerate(gen_data):
        if "response_to_q" not in task:
            continue
        existing = task.get("evaluation_error_analysis") or {}
        for metric, template in judge_prompts.items():
            if metric in existing and existing[metric].get("answer"):
                continue
            # acknow must come first because hallucinate depends on its extracted_pref
            prompt = template
            prompt = prompt.replace("{question}", task["question"])
            prompt = prompt.replace("{preference}", task["preference"])
            prompt = prompt.replace("{end_generation}", task["response_to_q"])
            if metric == "hallucinate":
                # will be filled in second pass once acknow is done
                continue
            pending.append((idx, metric, prompt))

    # Phase 1: acknow/violate/helpful in parallel
    print(f"[judge] phase 1: {len(pending)} prompts (acknow/violate/helpful)")
    results = judge.chat_batch(
        pending, build_messages=lambda it: [{"role": "user", "content": it[2]}],
        workers=args.workers, max_tokens=max_tokens,
        temperature=0.0, system=system_prompt,
        desc="PrefEval judge p1",
    )
    for (idx, metric, _prompt), out in zip(pending, results):
        text = "" if isinstance(out, Exception) else out
        if "evaluation_error_analysis" not in gen_data[idx]:
            gen_data[idx]["evaluation_error_analysis"] = {}
        entry: dict = {}
        if metric == "acknow":
            pref_extracted, answer = _parse_preference_answer(text)
            entry["answer"] = answer
            entry["extract_pref"] = pref_extracted
        else:
            expl, answer = _parse_explanation_answer(text)
            entry["explanation"] = expl
            entry["answer"] = answer
        gen_data[idx]["evaluation_error_analysis"][metric] = entry

    # Phase 2: hallucinate (depends on acknow.extract_pref)
    phase2: list[tuple[int, str]] = []  # (task_idx, prompt)
    template = judge_prompts["hallucinate"]
    for idx, task in enumerate(gen_data):
        ea = task.get("evaluation_error_analysis") or {}
        if "hallucinate" in ea and ea["hallucinate"].get("answer"):
            continue
        acknow = ea.get("acknow") or {}
        extracted_pref = acknow.get("extract_pref", "")
        prompt = template.replace("{preference}", task["preference"]).replace(
            "{assistant_restatement}", extracted_pref
        )
        phase2.append((idx, prompt))

    print(f"[judge] phase 2: {len(phase2)} prompts (hallucinate)")
    if phase2:
        results2 = judge.chat_batch(
            phase2, build_messages=lambda it: [{"role": "user", "content": it[1]}],
            workers=args.workers, max_tokens=max_tokens,
            temperature=0.0, system=system_prompt,
            desc="PrefEval judge p2",
        )
        for (idx, _prompt), out in zip(phase2, results2):
            text = "" if isinstance(out, Exception) else out
            expl, answer = _parse_explanation_answer(text)
            gen_data[idx].setdefault("evaluation_error_analysis", {})["hallucinate"] = {
                "explanation": expl, "answer": answer,
            }

    atomic_write_json(save_file, gen_data)
    print(f"[judge] saved → {save_file}")
    return save_file


# ---------------- stage: accuracy aggregation ----------------

def stage_accuracy(args) -> dict:
    judged = _paths(args, "judged_results")
    if not judged.exists():
        raise FileNotFoundError(f"No judged results at {judged} — run judge first.")
    with judged.open() as f:
        data = json.load(f)

    stats = {
        "acknowledgement": 0,
        "hallucination": 0,
        "violation": 0,
        "error_unhelpful": 0,
        "error_inconsistent": 0,
        "hallucination_of_preference_violation": 0,
        "preference_unaware_violation": 0,
        "preference_adherence_accuracy": 0,
    }
    n = 0
    for entry in data:
        ea = entry.get("evaluation_error_analysis") or {}
        if not ea:
            continue
        n += 1
        is_ack = "yes" in ea.get("acknow", {}).get("answer", "").lower()
        is_hallu = is_ack and "yes" in ea.get("hallucinate", {}).get("answer", "").lower()
        is_viol = "yes" in ea.get("violate", {}).get("answer", "").lower()
        is_unhelp = "no" in ea.get("helpful", {}).get("answer", "").lower()

        is_inconsistent = is_ack and not is_hallu and is_viol and not is_unhelp
        is_hallu_viol = is_ack and is_hallu and is_viol and not is_unhelp
        is_unaware = not is_ack and is_viol and not is_unhelp

        pfa = not any([is_inconsistent, is_hallu_viol, is_unaware, is_unhelp])

        stats["acknowledgement"] += int(is_ack)
        stats["hallucination"] += int(is_hallu)
        stats["violation"] += int(is_viol)
        stats["error_unhelpful"] += int(is_unhelp)
        stats["error_inconsistent"] += int(is_inconsistent)
        stats["hallucination_of_preference_violation"] += int(is_hallu_viol)
        stats["preference_unaware_violation"] += int(is_unaware)
        stats["preference_adherence_accuracy"] += int(pfa)

    summary = {
        "model": args.model, "topic": args.topic, "task": args.task,
        "inter_turns": args.inter_turns, "total": n,
        "preference_adherence_accuracy": (stats["preference_adherence_accuracy"] / n) if n else 0.0,
        "counts": stats,
    }
    out = _paths(args, "summary").with_suffix(".summary.json")
    atomic_write_json(out, summary)
    print(f"[accuracy] PAA={summary['preference_adherence_accuracy']:.2%} "
          f"on {n} items → {out}")
    return summary


# ---------------- CLI ----------------

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PrefEval unified runner.")
    add_common_llm_args(ap)
    ap.add_argument("--stage", default="all",
                    choices=["gen", "cls", "judge", "accuracy", "all"])
    ap.add_argument("--pref-form", "--pref_form", dest="pref_form",
                    default="explicit", choices=["explicit", "implicit"])
    ap.add_argument("--task", default="zero-shot",
                    choices=["zero-shot", "cot", "remind"],
                    help="(rag/selfcritic baselines not yet migrated)")
    ap.add_argument("--topic", required=True, help=f"One of: {ALL_TOPICS}")
    ap.add_argument("--inter-turns", "--inter_turns", dest="inter_turns",
                    type=int, default=2,
                    help="Number of filler conversation turns inserted mid-context.")
    # Judge-specific (optional): separate judge endpoint
    ap.add_argument("--judge-api-base", "--judge_api_base", dest="judge_api_base",
                    default=None, help="Optional separate base URL for the judge.")
    ap.add_argument("--judge-model", "--judge_model", dest="judge_model",
                    default=None, help="Judge model name if different from --model.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ensure_dir(args.output_dir)

    client = client_from_args(args)
    judge_client = client
    if args.judge_api_base or args.judge_model:
        judge_client = LLMClient(
            model_name=args.judge_model or args.model,
            api_base=args.judge_api_base or args.api_base,
            api_key=args.api_key,
            default_temperature=0.0,
            default_max_tokens=300,
            strip_think=True,
        )

    stages = ["gen", "cls", "judge", "accuracy"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "gen":
            stage_gen(client, args)
        elif stage == "cls":
            stage_cls(client, args)
        elif stage == "judge":
            stage_judge(client, args, judge_client=judge_client)
        elif stage == "accuracy":
            stage_accuracy(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
