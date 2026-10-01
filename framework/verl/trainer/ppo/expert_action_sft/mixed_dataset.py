"""Complete legacy episodes plus masked, real closed-loop Teacher episodes.

The stored native prompt and original Teacher response are immutable targets.
Context-only decisions remain in the certified source episode, not in forward
batches. Draw receipts retain their original indices rather than renumbering.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re

from .data import _require, _is_int, _text, TASK_TYPES, encode_record, validate_record

DATASET_FORMAT = "mixed_expert_episode_v1"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_native_response(response):
    """Same strict native protocol accepted by the collection quality gate."""
    match = re.fullmatch(r"\s*<think\s*>(.*?)</think\s*>\s*<action\s*>([^<>]+)</action\s*>\s*",
                         response, re.I | re.S)
    _require(match is not None, "Malformed original Teacher response")
    reasoning, action = (part.strip() for part in match.groups())
    _require(reasoning and action and "\n" not in action and "\r" not in action and
             not re.search(r"<\s*/?\s*(?:think|action)|<\||\[/?INST\]", reasoning, re.I),
             "Malformed original Teacher reasoning/action")
    return match, action


def encode_real_step(step, tokenizer, max_prompt_length, max_response_length, chat_kwargs=None):
    """Template stored native user prompt once; retain raw assistant bytes/EOS."""
    prompt = _text(step.get("prompt_text"), "native prompt")
    response = _text(step.get("response"), "original response")
    match, action = parse_native_response(response)
    _require(action.strip().lower() == str(step.get("action", "")).strip().lower(),
             "Original response differs from executed action")
    _require(getattr(tokenizer, "is_fast", False), "Native fast tokenizer required")
    kwargs = dict(chat_kwargs or {})
    _require(not ({"tokenize", "add_generation_prompt"} & set(kwargs)),
             "chat kwargs override native encoding controls")
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    full_text = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}],
                                             tokenize=False, add_generation_prompt=False, **kwargs)
    _require(full_text.startswith(prompt_text + response), "Native template altered original assistant content")
    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    full = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids, offsets = list(full["input_ids"]), full["offset_mapping"]
    _require(full_ids[:len(prompt_ids)] == prompt_ids, "Native prompt/response token boundary changed")
    response_ids = full_ids[len(prompt_ids):]
    eos = tokenizer.eos_token_id
    _require(eos is not None and eos in response_ids, "Native assistant EOS missing")
    response_ids = response_ids[:response_ids.index(eos) + 1]
    # Special-token offsets are sometimes (0, 0); verify by decoding the exact
    # target to prevent an embedded EOS from silently truncating original prose.
    decoded = tokenizer.decode(response_ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    _require(decoded == response, "Native response tokens do not preserve original response exactly")
    _require(0 < len(prompt_ids) <= max_prompt_length, "Native prompt exceeds budget; no truncation allowed")
    _require(0 < len(response_ids) <= max_response_length, "Native response exceeds budget; no truncation allowed")
    for key, count in (("prompt_tokens_native", len(prompt_ids)), ("response_tokens_native", len(response_ids))):
        if key in step:
            _require(type(step[key]) is int and step[key] == count, f"Recorded {key} differs from native encoding")
    begin, end = match.span(2)
    begin += len(prompt_text)
    end += len(prompt_text)
    # Diagnostic only: overlapping boundary tokens remain full-response targets.
    response_offsets = offsets[len(prompt_ids):len(prompt_ids) + len(response_ids)]
    action_mask = [int(left < end and right > begin and right > left)
                   for left, right in response_offsets]
    _require(any(action_mask), "Native response has no action token overlap")
    _require(_is_int(tokenizer.pad_token_id) and tokenizer.pad_token_id >= 0, "Native pad token missing")
    return {"prompt_ids": prompt_ids, "response_ids": response_ids, "action_mask": action_mask,
            "sft_mask": [1] * len(response_ids), "pad_token_id": tokenizer.pad_token_id}


def _legacy_rows(wrapper):
    rows, cert, source = wrapper["rows"], wrapper["certification"], wrapper["source_episode_id"]
    _require(isinstance(rows, list) and rows, "Empty legacy trajectory")
    ids = [f"{source}:{i}" for i in range(len(rows))]
    _require(cert["episode_id"] == source and cert["sample_ids"] == ids and
             cert["source_episode_won"] is True and cert["source_decisions"] == len(rows),
             "Legacy complete-trajectory certification mismatch")
    for index, row in enumerate(rows):
        validate_record(row)
        _require(row["sample_id"] == ids[index] and row["episode_id"] == source and row["step"] == index and
                 row["game_path"] == rows[0]["game_path"] and row["task"] == rows[0]["task"],
                 "Legacy trajectory identity/order mismatch")
        for earlier, transition in enumerate(row["expert_history"]):
            prior = rows[earlier]
            _require(transition == {"observation": prior["current_observation"], "action": prior["target_action"],
                                    "feedback": rows[earlier + 1]["current_observation"]},
                     "Legacy trajectory history is disconnected")
    return rows, list(range(len(rows)))


def _real_rows(wrapper):
    episode, source = wrapper["episode"], wrapper["source_episode_id"]
    _require(episode.get("schema") == "real_closedloop_episode_v1" and episode.get("episode_id") == source,
             "Closed-loop export identity/schema mismatch")
    raw, quality = episode["trajectory"], episode["quality"]
    _require(raw.get("episode_id") == source and quality.get("episode_id") == source and raw.get("won") is True and raw.get("done") is True and
             quality.get("accepted") is True, "Closed-loop episode is not an accepted success")
    game = PurePosixPath(_text(raw.get("game_path"), "game_path"))
    _require(not game.is_absolute() and ".." not in game.parts and game.parts[:2] == ("json_2.1.1", "train") and
             game.name == "game.tw-pddl" and raw.get("task_type") in TASK_TYPES, "Closed-loop source is not train")
    steps, flags = raw["steps"], quality["steps"]
    _require(isinstance(steps, list) and steps and len(steps) == raw.get("num_steps", len(steps)) and
             isinstance(flags, list) and len(flags) == len(steps), "Incomplete closed-loop episode/quality coverage")
    selected = []
    takeover = raw.get("takeover_at")
    is_takeover = raw.get("condition") == "student_takeover" or takeover is not None
    if is_takeover:
        _require(_is_int(takeover) and 0 <= takeover < len(steps), "Invalid takeover boundary")
        provenance = raw.get("source_provenance", {})
        _require(provenance.get("prefix_replay_verified") is True and
                 provenance.get("prefix_replay_steps") == takeover and
                 provenance.get("source_game_path") == raw["game_path"],
                 "Takeover prefix lacks matching verified replay provenance")
    for index, (step, flag) in enumerate(zip(steps, flags)):
        _require(type(step.get("step")) is int and step["step"] == index and flag.get("step") == index and
                 step.get("episode_id", source) == source, "Closed-loop step identity/order mismatch")
        _require(step.get("environment_executed") is True, "Closed-loop transition was not executed")
        for key in ("observation", "feedback", "prompt_text", "response", "action"):
            _text(step.get(key), key)
        _require(step.get("current_observation", step["observation"]) == step["observation"] and
                 step.get("after_observation", step["feedback"]) == step["feedback"], "Observation alias mismatch")
        if index:
            _require(steps[index - 1]["feedback"] == step["observation"] and
                     steps[index - 1].get("done") is not True, "Disconnected closed-loop history")
        if "prompt_sha256" in step:
            _require(step["prompt_sha256"] == hashlib.sha256(step["prompt_text"].encode()).hexdigest(),
                     "Native prompt receipt mismatch")
        _require(type(flag.get("supervise")) is bool and type(flag.get("sft_step_mask")) is int and
                 flag["sft_step_mask"] in (0, 1) and bool(flag["sft_step_mask"]) == flag["supervise"],
                 "Closed-loop supervision mask mismatch")
        # Early pure-Teacher accepted collections predate origin metadata. They
        # have no takeover boundary; never infer an origin in a takeover episode.
        origin = step.get("origin", "teacher" if not is_takeover else None)
        _require(origin in ("student_prefix", "teacher"), "Unknown closed-loop step origin")
        if is_takeover:
            _require(origin == ("student_prefix" if index < takeover else "teacher"), "Takeover origin/boundary mismatch")
        _require(origin != "student_prefix" or not flag["supervise"], "Student prefix cannot be supervised")
        if flag["supervise"]:
            _require(origin == "teacher" and isinstance(flag.get("semantic_review"), dict) and
                     flag["semantic_review"].get("verdict") == "accept" and
                     flag["semantic_review"].get("step") == index, "Supervised Teacher step lacks accepted review")
            _require(step.get("format_valid") is True and step.get("strict_format_valid") is True and
                     step.get("action_admissible") is True and step.get("finish_reason") == "stop" and
                     step.get("response_eos_included") is True and step.get("nothing_happens") is not True and
                     not re.search(r"\bnothing happens\b", step["feedback"], re.I), "Supervised Teacher step failed native checks")
            admissible = step.get("admissible_actions")
            _require(isinstance(admissible, list) and step["action"] in admissible, "Supervised action is not admissible")
            parse_native_response(step["response"])
            selected.append(index)
    _require(steps[-1].get("done") is True and steps[-1].get("won") is True, "Successful terminal transition missing")
    _require(selected, "Closed-loop episode has no supervised Teacher decisions")
    return steps, selected


class MixedExpertTrajectoryDataset:
    sampler_version = "sha256-uniform-mixed-complete-episode-with-replacement-v1"

    def __init__(self, config, tokenizer, chat_kwargs=None):
        _require(config.get("dataset_format") == DATASET_FORMAT, "Explicit mixed dataset format required")
        path = Path(config["dataset_path"]).expanduser().resolve(strict=True)
        content = path.read_bytes()
        self.dataset_path, self.dataset_sha256 = str(path), hashlib.sha256(content).hexdigest()
        _require(self.dataset_sha256 == config["dataset_sha256"], "Mixed dataset SHA256 mismatch")
        manifest_bytes = Path(config["trajectory_manifest_path"]).read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _require(self.manifest_sha256 == config["trajectory_manifest_sha256"], "Mixed manifest SHA256 mismatch")
        manifest = json.loads(manifest_bytes)
        _require(manifest.get("schema_version") == 2 and manifest.get("schema") == "mixed_expert_manifest_v1" and
                 manifest.get("dataset_sha256") == self.dataset_sha256, "Mixed manifest schema/dataset mismatch")
        entries = manifest["episodes"]
        _require(isinstance(entries, list) and entries, "Empty mixed manifest")
        self.episode_info = {entry["episode_id"]: entry for entry in entries}
        _require(len(self.episode_info) == len(entries), "Duplicate manifest episode")
        if path.suffix == ".gz":
            content = gzip.decompress(content)
        self.rows, self.encoded, self.episodes, self.source_episodes = [], [], {}, {}
        for line in content.decode("utf-8").splitlines():
            _require(bool(line.strip()), "Blank mixed dataset record")
            wrapper = json.loads(line)
            eid, source, kind = wrapper["episode_id"], wrapper["source_episode_id"], wrapper["kind"]
            _require(wrapper.get("schema") == DATASET_FORMAT and kind in ("legacy_complete", "real_closedloop"),
                     "Unknown mixed episode schema/kind")
            _require(eid == ("legacy:" if kind == "legacy_complete" else "closedloop:") + _text(source, "source id") and
                     eid not in self.episodes and eid in self.episode_info, "Mixed episode identity/coverage mismatch")
            info, digest = self.episode_info[eid], canonical_sha256(wrapper)
            rows, selected = _legacy_rows(wrapper) if kind == "legacy_complete" else _real_rows(wrapper)
            _require(info.get("source_episode_id") == source and info.get("kind") == kind and
                     info.get("source_episode_sha256") == digest and info.get("source_decisions") == len(rows) and
                     info.get("supervised_decisions") == len(selected) and info.get("selected_step_indices") == selected,
                     "Mixed manifest episode certification mismatch")
            indices = []
            for step in selected:
                args = (rows[step], tokenizer, int(config["max_prompt_length"]), int(config["max_response_length"]), chat_kwargs)
                encoded = encode_record(*args) if kind == "legacy_complete" else encode_real_step(*args)
                encoded.update(sample_id=f"{eid}:{step}", episode_id=eid, source_episode_id=source, source_step=step,
                               source_kind=kind, source_episode_sha256=digest,
                               sft_mask=[1] * len(encoded["response_ids"]))
                indices.append(len(self.encoded))
                self.encoded.append(encoded)
            self.rows.extend(rows)
            self.episodes[eid], self.source_episodes[eid] = indices, wrapper
        _require(set(self.episodes) == set(self.episode_info), "Missing mixed dataset episode")
        self.episode_ids = sorted(self.episodes)

    def sample_trajectories(self, seed, rollout_iteration, count):
        _require(_is_int(seed) and _is_int(rollout_iteration) and rollout_iteration >= 1 and
                 _is_int(count) and count >= 1, "Invalid mixed trajectory sampling request")
        material = f"{self.sampler_version}\0{self.dataset_sha256}\0{self.manifest_sha256}\0{seed}\0{rollout_iteration}".encode()
        derived = hashlib.sha256(material).hexdigest()
        rng, records, draws = random.Random(int(derived, 16)), [], []
        for index in range(count):
            eid = rng.choice(self.episode_ids)
            selected = copy.deepcopy([self.encoded[i] for i in self.episodes[eid]])
            info = self.episode_info[eid]
            draws.append({"draw": index, "episode_id": eid, "source_episode_id": info["source_episode_id"],
                          "source_kind": info["kind"], "source_episode_sha256": info["source_episode_sha256"],
                          "start": len(records), "decisions": len(selected), "total_decisions": info["source_decisions"],
                          "selected_step_indices": list(info["selected_step_indices"]),
                          "selected_sample_ids": [row["sample_id"] for row in selected],
                          "supervised_tokens": sum(sum(row["sft_mask"]) for row in selected)})
            records.extend(selected)
        return records, derived, draws
