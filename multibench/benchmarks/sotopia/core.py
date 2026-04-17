"""Sotopia simulation + 7-dim LLM judge — paper-aligned simplified harness.

Ported from the existing in-house `run_sotopia.py` which implements the
paper's evaluation protocol without depending on the full upstream sotopia
package (no Redis / FastAPI / agent database).

Semantics
---------
- 90 scenarios, 40 characters (upstream dataset, subsetted via
  `sotopia_scenarios.json`).
- Each scenario runs a two-agent round-robin interaction (up to 20 turns,
  stop after 2 consecutive "none" actions).
- Each dialogue is judged along 7 dimensions with per-dimension numeric
  ranges. Out-of-range scores are clamped to keep one bad-judge row from
  erroring the whole scenario.
- overall_score = mean of 7 dimension averages across all agents.

Parallelism: the whole benchmark is scenario-parallel via ThreadPoolExecutor
(each scenario's 20 sequential turns + 1 judge call runs on its own worker).
"""
from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ...client import LLMClient


# ---------- paper-defined dimensions ----------
DIMENSIONS = {
    "believability":                   {"range": [0, 10],  "type": "positive"},
    "relationship":                    {"range": [-5, 5],  "type": "bidirectional"},
    "knowledge":                       {"range": [0, 10],  "type": "positive"},
    "secret":                          {"range": [-10, 0], "type": "negative"},
    "social_rules":                    {"range": [-10, 0], "type": "negative"},
    "financial_and_material_benefits": {"range": [-5, 5],  "type": "bidirectional"},
    "goal":                            {"range": [0, 10],  "type": "positive"},
}


EVAL_PROMPT_TEMPLATE = """Please evaluate the social interaction between the agents on the following dimensions.

Scenario: {scenario}

{agent_backgrounds}

{agent_goals}

Conversation:
{conversation}

For EACH agent, evaluate on these 7 dimensions. Return a JSON object with the following structure:
{{
  "{agent1_name}": {{
    "believability": {{"reasoning": "...", "score": <int 0-10>}},
    "relationship": {{"reasoning": "...", "score": <int -5 to 5>}},
    "knowledge": {{"reasoning": "...", "score": <int 0-10>}},
    "secret": {{"reasoning": "...", "score": <int -10 to 0>}},
    "social_rules": {{"reasoning": "...", "score": <int -10 to 0>}},
    "financial_and_material_benefits": {{"reasoning": "...", "score": <int -5 to 5>}},
    "goal": {{"reasoning": "...", "score": <int 0-10>}}
  }},
  "{agent2_name}": {{
    "believability": {{"reasoning": "...", "score": <int 0-10>}},
    "relationship": {{"reasoning": "...", "score": <int -5 to 5>}},
    "knowledge": {{"reasoning": "...", "score": <int 0-10>}},
    "secret": {{"reasoning": "...", "score": <int -10 to 0>}},
    "social_rules": {{"reasoning": "...", "score": <int -10 to 0>}},
    "financial_and_material_benefits": {{"reasoning": "...", "score": <int -5 to 5>}},
    "goal": {{"reasoning": "...", "score": <int 0-10>}}
  }}
}}

Dimension guidelines:
- believability [0-10]: Is the agent's behavior natural, realistic, and consistent with their character?
- relationship [-5 to 5]: Did the interaction improve (+) or damage (-) the relationship?
- knowledge [0-10]: Did the agent gain new, important information?
- secret [-10 to 0]: Did the agent leak secrets? (0=no leaks, -10=critical leaks)
- social_rules [-10 to 0]: Did the agent violate social norms or laws? (0=acceptable, -10=law-breaking)
- financial_and_material_benefits [-5 to 5]: Did the agent gain (+) or lose (-) material benefits?
- goal [0-10]: How well did the agent achieve their social goals?

Respond with ONLY the JSON object."""


def clamp(v, low, high):
    if v > high:
        return high
    if v < low:
        return low
    return v


def validate_scores(scores_dict):
    """Return (valid, oob_dims): True only if every provided score is within
    its paper-defined range. This matches the upstream sotopia package's
    `zero_to_ten` / `minus_five_to_five` / `minus_ten_to_zero` validators
    which raise ValueError on out-of-bounds — we instead signal caller so
    it can retry before falling back to clamping.
    """
    oob = []
    for dim, info in DIMENSIONS.items():
        if dim not in scores_dict:
            continue
        raw = scores_dict[dim]
        low, high = info["range"]
        if raw < low or raw > high:
            oob.append((dim, raw, low, high))
    return (not oob), oob


def clamp_scores(scores_dict):
    """Fallback path — clamp OOB scores to the valid range. Missing dims
    default to 0 (negative/bidirectional) or 5 (positive, range midpoint)."""
    clamped = {}
    for dim, info in DIMENSIONS.items():
        if dim in scores_dict:
            raw = scores_dict[dim]
            low, high = info["range"]
            if info["type"] == "negative" and raw > 0:
                clamped[dim] = 0
            else:
                clamped[dim] = clamp(raw, low, high)
        else:
            if info["type"] == "negative":
                clamped[dim] = 0
            elif info["type"] == "bidirectional":
                clamped[dim] = 0
            else:
                clamped[dim] = 5
    return clamped


def _clean_goal(g: str) -> str:
    g = re.sub(r"<extra_info>.*?</extra_info>", "", g, flags=re.DOTALL)
    g = re.sub(r"<strategy_hint>.*?</strategy_hint>", "", g, flags=re.DOTALL)
    return g.strip()


def simulate_interaction(
    scenario: dict,
    *,
    sim_client: LLMClient,
    max_turns: int = 20,
    max_stale: int = 2,
    temperature: float = 1.0,
):
    agents = list(scenario["agents_background"].keys())
    agent1_name, agent2_name = agents[0], agents[1]
    agent1_bg = scenario["agents_background"][agent1_name]
    agent2_bg = scenario["agents_background"][agent2_name]
    goal1 = _clean_goal(scenario["social_goals"].get(agent1_name, ""))
    goal2 = _clean_goal(scenario["social_goals"].get(agent2_name, ""))
    scene = scenario["scenario"]

    agent1_system = (
        f"You are roleplaying as {agent1_name} in the following scenario.\n\n"
        f"Background: {agent1_bg}\n\nScenario: {scene}\n\n"
        f"Your private goal: {goal1}\n\n"
        "At each turn you can choose one action:\n"
        "- speak(\"your message\")\n"
        "- non-verbal communication(\"your gesture/expression\")\n"
        "- action(\"your physical action\")\n"
        "- leave (to end the conversation)\n"
        "- none (do nothing)\n\n"
        "Respond with ONLY one action. Keep responses natural and in character."
    )
    agent2_system = (
        f"You are roleplaying as {agent2_name} in the following scenario.\n\n"
        f"Background: {agent2_bg}\n\nScenario: {scene}\n\n"
        f"Your private goal: {goal2}\n\n"
        "At each turn you can choose one action:\n"
        "- speak(\"your message\")\n"
        "- non-verbal communication(\"your gesture/expression\")\n"
        "- action(\"your physical action\")\n"
        "- leave (to end the conversation)\n"
        "- none (do nothing)\n\n"
        "Respond with ONLY one action. Keep responses natural and in character."
    )

    conversation = []
    stale_count = 0
    for turn in range(max_turns):
        if turn % 2 == 0:
            speaker_name, speaker_system = agent1_name, agent1_system
        else:
            speaker_name, speaker_system = agent2_name, agent2_system
        if not conversation:
            user_msg = f"The scene begins. You are {speaker_name}. Start the interaction."
        else:
            conv_text = "\n".join(f"{c['speaker']}: {c['action']}" for c in conversation)
            user_msg = f"Conversation so far:\n{conv_text}\n\nIt's your turn. Respond as {speaker_name}."
        try:
            response = sim_client.chat(
                [{"role": "user", "content": user_msg}],
                system=speaker_system, temperature=temperature, max_tokens=256,
            )
        except Exception:
            response = "none"
        if not response:
            response = "none"
        rl = response.strip().lower()
        if "leave" in rl and len(rl) < 30:
            conversation.append({"speaker": speaker_name, "action": response.strip(), "turn": turn})
            break
        if rl.strip() in ("none", "none()"):
            stale_count += 1
            if stale_count >= max_stale:
                break
            continue
        stale_count = 0
        conversation.append({"speaker": speaker_name, "action": response.strip(), "turn": turn})
    return conversation, agents


def _parse_judge_response(response: str, agents: list) -> dict:
    """Parse the judge's JSON-object response into per-agent raw score dicts."""
    if not response:
        return {}
    clean = response.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 2:
            clean = parts[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        try:
            import json_repair
            parsed = json_repair.loads(clean)
        except Exception:
            parsed = {}
    if not isinstance(parsed, dict):
        return {}

    per_agent = {}
    for agent_name in agents:
        data = parsed.get(agent_name)
        if data is None:
            for key in parsed:
                if isinstance(key, str) and agent_name.split()[0].lower() in key.lower():
                    data = parsed[key]
                    break
        raw = {}
        if isinstance(data, dict):
            for dim in DIMENSIONS:
                v = data.get(dim)
                if isinstance(v, dict) and "score" in v:
                    raw[dim] = v["score"]
                elif isinstance(v, (int, float)):
                    raw[dim] = v
        per_agent[agent_name] = raw
    return per_agent


def evaluate_interaction(
    scenario: dict, conversation: list, agents: list, *,
    judge_client: LLMClient, max_tokens: int = 2048,
    max_judge_retries: int = 1,
):
    """Judge the 7 paper dimensions per agent.

    Protocol alignment with upstream sotopia:
      - Upstream validators raise ValueError on OOB scores. We instead retry
        the judge up to `max_judge_retries` times on OOB and only clamp as a
        final fallback. This preserves the paper's "bad judge output is
        invalid" semantics while staying robust at scale.
      - Every scenario records `oob_events` so aggregators can track judge
        quality over a run.
    """
    import logging
    agent1_name, agent2_name = agents[0], agents[1]
    conv_text = "\n".join(f"Turn {c['turn']}: {c['speaker']}: {c['action']}" for c in conversation)
    bg_text = "".join(f"{n}: {b}\n\n" for n, b in scenario["agents_background"].items())
    goals_text = ""
    for n, g in scenario["social_goals"].items():
        goals_text += f"{n}'s goal: {_clean_goal(g)}\n"

    prompt = EVAL_PROMPT_TEMPLATE.format(
        scenario=scenario["scenario"],
        agent_backgrounds=bg_text, agent_goals=goals_text,
        conversation=conv_text,
        agent1_name=agent1_name, agent2_name=agent2_name,
    )

    raw_per_agent: dict = {}
    oob_events: list = []
    for attempt in range(max_judge_retries + 1):
        try:
            response = judge_client.chat(prompt, temperature=0.0, max_tokens=max_tokens)
        except Exception:
            response = ""
        raw_per_agent = _parse_judge_response(response, agents)
        any_oob = False
        for agent_name in agents:
            _, oob = validate_scores(raw_per_agent.get(agent_name, {}))
            if oob:
                any_oob = True
                oob_events.extend([(agent_name, *o, attempt) for o in oob])
        if not any_oob:
            break  # valid scores → done
        if attempt < max_judge_retries:
            logging.warning(
                "sotopia judge produced OOB scores (attempt %d): %s — retrying",
                attempt + 1, oob_events[-5:],
            )

    # Final: clamp whatever we ended up with
    agent_scores = {n: clamp_scores(raw_per_agent.get(n, {})) for n in agents}
    for agent_name in agents:
        dim_values = [agent_scores[agent_name][d] for d in DIMENSIONS if d in agent_scores[agent_name]]
        agent_scores[agent_name]["overall_score"] = (
            sum(dim_values) / len(dim_values) if dim_values else 0.0
        )
    # Expose OOB events so the aggregator can surface judge quality
    if oob_events:
        agent_scores["_meta"] = {"oob_events": oob_events}
    return agent_scores


def _process_scenario(idx, scenario, sim_client: LLMClient, judge_client: LLMClient):
    conversation, agents = simulate_interaction(scenario, sim_client=sim_client)
    agent_scores = evaluate_interaction(
        scenario, conversation, agents, judge_client=judge_client,
    )
    return {
        "scenario_idx": idx,
        "environment_id": scenario.get("environment_id", ""),
        "scenario": scenario["scenario"],
        "agents": agents,
        "conversation": conversation,
        "agent_scores": agent_scores,
    }


def run_sotopia(
    scenarios: list,
    *,
    sim_client: LLMClient,
    judge_client: LLMClient,
    workers: int = 16,
    seed: int = 42,
    max_scenarios: int | None = None,
) -> tuple[list, dict]:
    if max_scenarios:
        random.seed(seed)
        scenarios = random.sample(scenarios, min(max_scenarios, len(scenarios)))

    results = [None] * len(scenarios)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(_process_scenario, i, s, sim_client, judge_client): i
            for i, s in enumerate(scenarios)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sotopia"):
            i = futures[fut]
            results[i] = fut.result()

    all_agent_scores = []
    total_oob_events = 0
    scenarios_with_oob = 0
    for r in results:
        if r is None:
            continue
        for name in r["agents"]:
            all_agent_scores.append(r["agent_scores"][name])
        meta = r["agent_scores"].get("_meta", {})
        if meta.get("oob_events"):
            total_oob_events += len(meta["oob_events"])
            scenarios_with_oob += 1

    summary = {
        "model": sim_client.model_name,
        "judge_model": judge_client.model_name,
        "total_scenarios": len(scenarios),
        "total_agent_evaluations": len(all_agent_scores),
        "judge_oob_events": total_oob_events,
        "scenarios_with_oob": scenarios_with_oob,
        "dimensions": {},
    }
    for dim in DIMENSIONS:
        scores = [s.get(dim, 0) for s in all_agent_scores]
        avg = sum(scores) / len(scores) if scores else 0
        std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0
        summary["dimensions"][dim] = {
            "range": DIMENSIONS[dim]["range"],
            "avg": round(avg, 4), "std": round(std, 4), "count": len(scores),
        }
    dim_avgs = [summary["dimensions"][d]["avg"] for d in DIMENSIONS]
    summary["overall_score"] = round(sum(dim_avgs) / len(dim_avgs), 4) if dim_avgs else 0.0
    per_agent_overalls = [s.get("overall_score", 0) for s in all_agent_scores]
    summary["per_agent_overall_avg"] = round(
        sum(per_agent_overalls) / len(per_agent_overalls), 4,
    ) if per_agent_overalls else 0.0
    return results, summary
