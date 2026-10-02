"""Validate, encode and sample independent offline expert examples.

Auxiliary examples travel in meta_info as one global batch. Student tensors,
grouping and per-row metadata remain native. Trajectory draws have explicit
occurrence boundaries, including when sampling with replacement repeats one.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import tempfile


SCHEMA_VERSION = 1
SAMPLER_VERSION = "sha256-local-random-sample-v1"
TASK_TYPES = frozenset({
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
})
ROW_FIELDS = frozenset({
    "sample_id", "episode_id", "game_path", "task_type", "task", "step",
    "expert_history", "current_observation", "admissible_actions", "prompt_text",
    "prompt_history_length", "target_action", "target_action_token_ids",
    "reasoning_target", "source_episode_won", "reasoning", "reasoning_source",
})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def _hex_digest(value, name):
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), f"invalid {name}")


def validate_record(row):
    """Check mechanical provenance, never claim this proves prose grounding."""
    _require(isinstance(row, dict), "expert row must be an object")
    optional_provenance = {"source_game_sha256", "source_dataset_sha256"}
    unknown = set(row) - ROW_FIELDS - optional_provenance
    missing = ROW_FIELDS - set(row)
    _require(not unknown, f"unknown expert fields (possible hidden/future data): {sorted(unknown)}")
    _require(not missing, f"missing expert fields: {sorted(missing)}")
    sid = _text(row["sample_id"], "sample_id")
    episode = _text(row["episode_id"], "episode_id")
    step = row["step"]
    _require(_is_int(step) and step >= 0, f"{sid}: invalid decision step")
    _require(sid == f"{episode}:{step}", f"{sid}: sample identity disagrees with episode/step")
    game = PurePosixPath(_text(row["game_path"], "game_path"))
    _require(not game.is_absolute() and ".." not in game.parts and
             game.parts[:2] == ("json_2.1.1", "train") and game.name == "game.tw-pddl",
             f"{sid}: source is not an ALFWorld training game")
    _require(row["task_type"] in TASK_TYPES, f"{sid}: unknown task type")
    _require(row["source_episode_won"] is True, f"{sid}: source episode was not successful")
    _require(row["reasoning_target"] is None,
             f"{sid}: legacy reasoning_target metadata must be null; the run selects the supervision mask")
    _require(row["prompt_history_length"] == 2, f"{sid}: native history length changed")
    prompt = _text(row["prompt_text"], "prompt_text")
    task = _text(row["task"], "task")
    observation = _text(row["current_observation"], "current_observation")
    _require(task in prompt and observation in prompt, f"{sid}: prompt missing task/current observation")
    history = row["expert_history"]
    _require(isinstance(history, list) and len(history) == step, f"{sid}: history length disagrees with step")
    for index, transition in enumerate(history):
        _require(isinstance(transition, dict) and set(transition) == {"observation", "action", "feedback"},
                 f"{sid}: invalid visible history transition {index}")
        for key, value in transition.items():
            _text(value, f"{sid}.history[{index}].{key}")
        if index:
            _require(history[index - 1]["feedback"] == transition["observation"],
                     f"{sid}: disconnected expert history")
    if history:
        _require(history[-1]["feedback"] == observation, f"{sid}: current state differs from expert history")
    action = _text(row["target_action"], "target_action")
    _require(action == action.strip() and not any(c in action for c in "\r\n<>"),
             f"{sid}: action must be the original bare command")
    admissible = row["admissible_actions"]
    _require(isinstance(admissible, list) and all(isinstance(x, str) for x in admissible) and
             action in admissible, f"{sid}: action is not admissible in its expert state")
    target_ids = row["target_action_token_ids"]
    _require(isinstance(target_ids, list) and target_ids and all(_is_int(x) and x >= 0 for x in target_ids),
             f"{sid}: invalid original target token IDs")
    reasoning = _text(row["reasoning"], "reasoning")
    _require(not any(marker in reasoning.lower() for marker in
                     ("<think", "</think", "<action", "</action", "<|", "[inst]", "[/inst]")),
             f"{sid}: reasoning contains protocol/control tags")
    source = row["reasoning_source"]
    _require(isinstance(source, dict) and
             set(source) == {"kind", "context_scope", "annotator", "version"},
             f"{sid}: missing reasoning provenance")
    _require(source["kind"] in {"manual_visible_state", "llm_visible_state"} and
             source["context_scope"] == "current_and_past_visible_only",
             f"{sid}: reasoning uses an unsupported source/context scope")
    _text(source["annotator"], "reasoning_source.annotator")
    _text(source["version"], "reasoning_source.version")
    return row


def _decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def encode_record(row, tokenizer, max_prompt_length, max_response_length, chat_kwargs=None):
    """Encode full native chat, rejecting any token crossing the Action boundary."""
    validate_record(row)
    sid = row["sample_id"]
    _require(getattr(tokenizer, "is_fast", False), "exact Action offsets require the native fast tokenizer")
    kwargs = dict(chat_kwargs or {})
    _require(not ({"tokenize", "add_generation_prompt"} & set(kwargs)),
             "chat template kwargs override native encoding controls")
    # Native rollout_loop.py constructs a user-only chat.  Qwen's template itself
    # inserts its default system message; manually adding another would differ.
    messages = [{"role": "user", "content": row["prompt_text"]}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
    assistant = "<think>\n" + row["reasoning"] + "\n</think>\n<action>" + row["target_action"] + "</action>"
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": assistant}],
        add_generation_prompt=False, tokenize=False, **kwargs,
    )
    _require(full_text.startswith(prompt_text + assistant), f"{sid}: native template changed assistant prefix/content")
    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    full = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = list(full["input_ids"])
    offsets = [tuple(pair) for pair in full["offset_mapping"]]
    _require(len(full_ids) == len(offsets), f"{sid}: token/offset count differs")
    _require(full_ids[:len(prompt_ids)] == prompt_ids, f"{sid}: token crosses prompt/response boundary")
    _require(_decode(tokenizer, row["target_action_token_ids"]) == row["target_action"],
             f"{sid}: original expert target IDs do not match this tokenizer")
    action_start = len(prompt_text) + len("<think>\n" + row["reasoning"] + "\n</think>\n<action>")
    action_end = action_start + len(row["target_action"])
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    _require(_is_int(eos) and _is_int(pad), "native tokenizer must provide EOS and padding IDs")
    # The completed chat adds the native end-of-message token and often a newline.
    # Keep EOS as context, with zero CE mask, exactly as a finished rollout does.
    end_candidates = [i for i in range(len(prompt_ids), len(full_ids))
                      if full_ids[i] == eos and offsets[i][0] >= action_end]
    _require(bool(end_candidates), f"{sid}: native completed assistant has no EOS")
    stop = end_candidates[0] + 1
    response_ids = full_ids[len(prompt_ids):stop]
    mask = []
    covered = []
    for token, (start, end) in zip(response_ids, offsets[len(prompt_ids):stop]):
        overlap = start < action_end and end > action_start
        if overlap:
            _require(start >= action_start and end <= action_end and end > start,
                     f"{sid}: token crosses Action boundary at {(start, end)}")
            _require(token != eos, f"{sid}: EOS overlaps action content")
            covered.append((start, end))
        mask.append(int(overlap))
    _require(bool(covered) and covered[0][0] == action_start and covered[-1][1] == action_end,
             f"{sid}: Action span is not completely represented")
    _require(all(a[1] == b[0] for a, b in zip(covered, covered[1:])), f"{sid}: gap/overlap in Action token offsets")
    action_ids = [token for token, selected in zip(response_ids, mask) if selected]
    _require(_decode(tokenizer, action_ids) == row["target_action"], f"{sid}: supervised IDs change original action")
    _require(0 < len(prompt_ids) <= max_prompt_length, f"{sid}: prompt exceeds configured limit; truncation forbidden")
    _require(0 < len(response_ids) <= max_response_length, f"{sid}: response exceeds configured limit; truncation forbidden")
    _require(mask[-1] == 0 and response_ids[-1] == eos, f"{sid}: EOS must have zero SFT mask")
    return {
        "sample_id": sid, "prompt_ids": prompt_ids, "response_ids": response_ids,
        "action_mask": mask, "pad_token_id": pad,
    }


class ExpertActionDataset:
    def __init__(self, dataset_path, dataset_sha256, tokenizer, max_prompt_length, max_response_length, chat_kwargs=None):
        path = Path(dataset_path).expanduser().resolve(strict=True)
        content = path.read_bytes()
        if dataset_sha256:
            _hex_digest(dataset_sha256, "dataset_sha256")
            _require(_sha256(content) == dataset_sha256, "Optional expert dataset checksum mismatch")
        text = gzip.decompress(content).decode("utf-8") if path.suffix == ".gz" else content.decode("utf-8")
        self.rows = [validate_record(json.loads(line)) for line in text.splitlines() if line.strip()]
        _require(len(self.rows) >= 4, "expert library needs at least four annotated decisions")
        _require(len({row["sample_id"] for row in self.rows}) == len(self.rows), "duplicate expert sample IDs")
        self.dataset_sha256 = dataset_sha256 or _sha256(content)
        self.dataset_path = str(path)
        # Encode and validate every annotated example once, before any update.
        # A malformed unsampled row must not surface halfway into a training run.
        self.encoded = [encode_record(row, tokenizer, max_prompt_length, max_response_length, chat_kwargs)
                        for row in self.rows]

    def sample(self, seed, rollout_iteration):
        _require(_is_int(seed), "SFT seed must be an integer")
        _require(_is_int(rollout_iteration) and rollout_iteration >= 1, "invalid rollout iteration")
        material = f"{SAMPLER_VERSION}\0{self.dataset_sha256}\0{seed}\0{rollout_iteration}".encode("utf-8")
        derived_seed = _sha256(material)
        indices = random.Random(int(derived_seed, 16)).sample(range(len(self.rows)), 4)
        return copy.deepcopy([self.encoded[index] for index in indices]), derived_seed


def _atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(json.loads(path.read_text(encoding="utf-8")) == content,
                 f"existing SFT contract/sample receipt disagrees: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".expert-sft-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(content))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_expert_action_sft(trainer, batch, rollout_iteration):
    """Attach one global batch of four independent experts before update_actor."""
    actor = trainer.config.actor_rollout_ref.actor
    config = actor.get("expert_action_sft", {})
    if not config.get("enabled", False):
        return {}
    if config.get("mode") == "complete_trajectory_reasoning_action":
        return prepare_trajectory_sft(trainer, batch, rollout_iteration)
    coef = float(config["coef"])
    _require(math.isfinite(coef) and coef >= 0, "invalid expert SFT coefficient")
    if coef == 0:
        return {"expert_sft/enabled": 0.0, "expert_sft/coef": 0.0}
    _require(config["global_samples"] == 4, "first expert Action SFT version requires total four decisions")
    seed = config["seed"]
    _require(_is_int(seed), "SFT seed must be an integer")
    chat_kwargs = dict(trainer.config.data.get("apply_chat_template_kwargs", {}))
    contract = {
        "schema_version": SCHEMA_VERSION, "sampler_version": SAMPLER_VERSION,
        "dataset_sha256": config.get("dataset_sha256"), "seed": seed,
        "global_samples": 4, "coef": coef,
        "max_prompt_length": int(config["max_prompt_length"]),
        "max_response_length": int(config["max_response_length"]),
        "chat_template_sha256": _sha256(str(trainer.tokenizer.chat_template).encode("utf-8")),
        "chat_kwargs": chat_kwargs,
        "supervised_region": "original_expert_action_content_only",
        "reasoning_context": "offline_nonempty_visible_state_reasoning",
        "rng": "independent_stateless_per_rollout",
    }
    _require(contract["max_prompt_length"] > 0 and contract["max_response_length"] > 0, "invalid SFT length limits")
    previous = getattr(trainer, "_expert_action_sft_contract", None)
    if previous is None:
        dataset = ExpertActionDataset(
            config["dataset_path"], config.get("dataset_sha256"), trainer.tokenizer,
            contract["max_prompt_length"], contract["max_response_length"], chat_kwargs,
        )
        trainer._expert_action_sft_dataset = dataset
        trainer._expert_action_sft_contract = contract
    else:
        _require(previous == contract, "expert SFT settings changed within the run")
        dataset = trainer._expert_action_sft_dataset
        _require(str(Path(config["dataset_path"]).expanduser().resolve(strict=True)) == dataset.dataset_path,
                 "expert dataset path changed within the run")
    checkpoint_root = Path(trainer.config.trainer.default_local_dir).expanduser().resolve()
    report_dir = (checkpoint_root / "expert_action_sft_sampling").resolve()
    _require(report_dir.is_relative_to(checkpoint_root), "SFT receipt directory escapes run root")
    _atomic_json(report_dir / "contract.json", contract)
    records, derived_seed = dataset.sample(seed, rollout_iteration)
    payload = {
        "schema_version": SCHEMA_VERSION, "rollout_iteration": rollout_iteration,
        "global_samples": 4, "coef": coef, "records": records,
    }
    _require("expert_action_sft" not in batch.meta_info, "expert SFT payload was already attached")
    evidence = {
        "schema_version": SCHEMA_VERSION, "stage": "prepared_before_actor_update",
        "rollout_iteration": rollout_iteration, "seed": seed,
        "derived_seed_sha256": derived_seed, "dataset_sha256": dataset.dataset_sha256,
        "global_samples": 4, "coef": coef,
        "sample_ids_in_rank_order": [record["sample_id"] for record in records],
        "samples": [{
            "sample_id": record["sample_id"], "prompt_tokens": len(record["prompt_ids"]),
            "response_tokens": len(record["response_ids"]), "action_tokens": sum(record["action_mask"]),
            "prompt_text": _decode(trainer.tokenizer, record["prompt_ids"]),
            "response_text": _decode(trainer.tokenizer, record["response_ids"]),
            "action_token_ids": [token for token, selected in zip(record["response_ids"], record["action_mask"]) if selected],
            "action_mask": record["action_mask"],
        } for record in records],
    }
    _atomic_json(report_dir / f"rollout_{rollout_iteration:06d}.json", evidence)
    batch.meta_info["expert_action_sft"] = payload
    return {
        "expert_sft/enabled": 1.0, "expert_sft/coef": coef,
        "expert_sft/global_samples": 4, "expert_sft/library_decisions": len(dataset.rows),
        "expert_sft/action_tokens": sum(sum(record["action_mask"]) for record in records),
        "expert_sft/input_tokens": sum(len(record["prompt_ids"]) + len(record["response_ids"]) for record in records),
    }


class ExpertTrajectoryDataset(ExpertActionDataset):
    """Only independently certified, complete successful training episodes."""
    sampler_version = "sha256-uniform-complete-episode-with-replacement-v1"

    def __init__(self, config, tokenizer, chat_kwargs=None):
        super().__init__(config["dataset_path"], config.get("dataset_sha256"), tokenizer,
                         int(config["max_prompt_length"]), int(config["max_response_length"]), chat_kwargs)
        manifest_content = Path(config["trajectory_manifest_path"]).read_bytes()
        if config.get("trajectory_manifest_sha256"):
            _require(_sha256(manifest_content) == config["trajectory_manifest_sha256"],
                     "Optional complete-trajectory manifest checksum mismatch")
        manifest = json.loads(manifest_content)
        _require(manifest["schema_version"] == 1, "Unknown complete-trajectory manifest schema")
        by_id = {row["sample_id"]: index for index, row in enumerate(self.rows)}
        self.episodes = {}
        used = set()
        for episode in manifest["episodes"]:
            eid, ids = episode["episode_id"], episode["sample_ids"]
            _require(eid not in self.episodes and ids and ids == [f"{eid}:{i}" for i in range(len(ids))],
                     "Duplicate or noncontiguous expert trajectory")
            _require(episode["source_episode_won"] is True and episode["source_decisions"] == len(ids),
                     "Expert trajectory does not contain every source decision")
            _require(all(sid in by_id and sid not in used for sid in ids), "Missing/duplicate expert decision")
            indices = [by_id[sid] for sid in ids]
            rows = [self.rows[index] for index in indices]
            first = rows[0]
            for step, row in enumerate(rows):
                _require(row["step"] == step and row["episode_id"] == eid and
                         row["game_path"] == first["game_path"] and row["task"] == first["task"],
                         "Expert decision belongs to another trajectory/task")
                for earlier, transition in enumerate(row["expert_history"]):
                    prior = rows[earlier]
                    _require(transition["observation"] == prior["current_observation"] and
                             transition["action"] == prior["target_action"] and
                             transition["feedback"] == rows[earlier + 1]["current_observation"],
                             "Expert trajectory history/action/state is disconnected")
                encoded = self.encoded[indices[step]]
                # Full native assistant response, including reasoning, tags and EOS.
                # action_mask remains unchanged and can still audit exact action IDs.
                encoded["sft_mask"] = [1] * len(encoded["response_ids"])
            used.update(ids)
            self.episodes[eid] = indices
        _require(bool(self.episodes) and len(used) == len(self.rows),
                 "Training dataset contains uncertified or incomplete trajectories")
        self.episode_ids = sorted(self.episodes)
        self.manifest_sha256 = _sha256(manifest_content)

    def sample(self, seed, rollout_iteration):
        _require(_is_int(seed) and _is_int(rollout_iteration) and rollout_iteration >= 1, "Invalid SFT sample seed/iteration")
        material = f"{self.sampler_version}\0{self.dataset_sha256}\0{self.manifest_sha256}\0{seed}\0{rollout_iteration}".encode()
        derived = _sha256(material)
        eid = random.Random(int(derived, 16)).choice(self.episode_ids)
        return copy.deepcopy([self.encoded[i] for i in self.episodes[eid]]), derived, eid

    def sample_trajectories(self, seed, rollout_iteration, count):
        """Uniform independent draws with replacement; no native RNG consumed."""
        _require(_is_int(seed) and _is_int(rollout_iteration) and rollout_iteration >= 1,
                 "Invalid SFT sample seed/iteration")
        _require(_is_int(count) and count >= 1, "Invalid number of expert trajectories")
        material = f"{self.sampler_version}\0{self.dataset_sha256}\0{self.manifest_sha256}\0{seed}\0{rollout_iteration}".encode()
        derived = _sha256(material)
        rng = random.Random(int(derived, 16))
        records, draws = [], []
        for draw in range(count):
            eid = rng.choice(self.episode_ids)
            selected = copy.deepcopy([self.encoded[i] for i in self.episodes[eid]])
            draws.append({"draw": draw, "episode_id": eid, "start": len(records),
                          "decisions": len(selected),
                          "supervised_tokens": sum(sum(row["sft_mask"]) for row in selected)})
            records.extend(selected)
        return records, derived, draws


def prepare_trajectory_sft(trainer, batch, rollout_iteration):
    from verl.trainer.ppo.expert_action_sft.schedule import MODE, coefficient
    config = trainer.config.actor_rollout_ref.actor["expert_action_sft"]
    coef = coefficient(config, rollout_iteration)
    trajectories = config.get("trajectories_per_round", 1)
    reduction = config.get("loss_reduction", "sequence-mean")
    _require(_is_int(trajectories) and trajectories >= 1, "Invalid trajectories_per_round")
    _require(reduction in ("sequence-mean", "token-mean"), "Invalid expert SFT loss reduction")
    token_mean = reduction == "token-mean"
    dataset_format = config.get("dataset_format", "legacy_complete_decisions_v1")
    _require(dataset_format in ("legacy_complete_decisions_v1", "mixed_expert_episode_v1"),
             "Unknown expert trajectory dataset format")
    mixed = dataset_format == "mixed_expert_episode_v1"
    _require(not mixed or token_mean, "Mixed complete trajectories require global token-mean")
    _require(token_mean or trajectories == 1, "Multiple trajectories require explicit global token-mean")
    _require("global_samples" not in config, "Complete trajectories have variable length; remove fixed global_samples")
    _require("expert_action_sft" not in batch.meta_info, "Expert SFT payload was already attached")
    payload = {"schema_version": 4 if mixed else 3 if token_mean else 2, "mode": MODE, "rollout_iteration": rollout_iteration,
               "coef": coef, "global_samples": 0, "records": []}
    if mixed:
        payload["dataset_format"] = dataset_format
    if token_mean:
        payload.update(loss_reduction="token-mean", trajectory_count=0, trajectory_draws=[], global_supervised_tokens=0)
    # The actor independently validates the counter and zero coefficient. No
    # loading, sampling or forward/backward once the auxiliary term has decayed.
    if coef == 0:
        batch.meta_info["expert_action_sft"] = payload
        return {"expert_sft/enabled": 0.0, "expert_sft/coef": 0.0, "expert_sft/global_samples": 0,
                "expert_sft/trajectories": 0, "expert_sft/supervised_tokens": 0}
    chat_kwargs = dict(trainer.config.data.get("apply_chat_template_kwargs", {}))
    dataset_class = ExpertTrajectoryDataset
    if mixed:
        from verl.trainer.ppo.expert_action_sft.mixed_dataset import MixedExpertTrajectoryDataset
        dataset_class = MixedExpertTrajectoryDataset
    contract = {
        "schema_version": payload["schema_version"], "mode": MODE, "sampler_version": dataset_class.sampler_version,
        "dataset_format": dataset_format,
        "seed": config["seed"], "initial_coef": float(config["coef"]),
        "decay_end_iteration": config["decay_end_iteration"], "decay_schedule": config.get("decay_schedule", "linear"),
        "schedule_counter": "one_based_outer_rollout_iteration; initial_at_1; zero_at_end",
        "max_prompt_length": int(config["max_prompt_length"]), "max_response_length": int(config["max_response_length"]),
        "chat_template_sha256": _sha256(str(trainer.tokenizer.chat_template).encode()), "chat_kwargs": chat_kwargs,
        "supervised_region": "complete_assistant_reasoning_action_protocol_eos",
        "normalization": "global_supervised_token_mean" if token_mean else "mean_of_decision_token_means_within_one_trajectory",
        "trajectories_per_round": trajectories,
        "attachment": "final_native_optimizer_update_once_per_outer_iteration",
        "rng": "independent_stateless_per_rollout_uniform_episode_with_replacement",
    }
    if mixed:
        contract.update(dataset_format=dataset_format,
                        supervised_region="approved_teacher_original_response_reasoning_action_protocol_eos;legacy_unchanged",
                        context_only_steps="preserved_in_full_source_episode;excluded_from_forward;original_indices_retained")
    prior = getattr(trainer, "_expert_trajectory_sft_contract", None)
    if prior is None:
        dataset = dataset_class(config, trainer.tokenizer, chat_kwargs)
        trainer._expert_trajectory_sft_dataset = dataset
        trainer._expert_trajectory_sft_contract = contract
    else:
        _require(prior == contract, "Expert trajectory SFT settings changed within run")
        dataset = trainer._expert_trajectory_sft_dataset
        _require(str(Path(config["dataset_path"]).resolve(strict=True)) == dataset.dataset_path, "Expert dataset path changed")
    if token_mean:
        records, derived, draws = dataset.sample_trajectories(config["seed"], rollout_iteration, trajectories)
        payload.update(global_samples=len(records), records=records, trajectory_count=len(draws),
                       trajectory_draws=draws, global_supervised_tokens=sum(sum(row["sft_mask"]) for row in records))
    else:
        records, derived, episode = dataset.sample(config["seed"], rollout_iteration)
        payload.update(global_samples=len(records), records=records, episode_id=episode)
    checkpoint_root = Path(trainer.config.trainer.default_local_dir).expanduser().resolve()
    receipt_root = checkpoint_root / "expert_action_sft_sampling"
    _atomic_json(receipt_root / "contract.json", contract)
    _atomic_json(receipt_root / f"rollout_{rollout_iteration:06d}.json", {
        "stage": "prepared_before_actor_update", "dataset_id": getattr(dataset, "dataset_id", "expert-trajectories"),
        "sampling_seed": derived,
        **payload,
        "decoded_samples": [{"sample_id": row["sample_id"],
                             "prompt": _decode(trainer.tokenizer, row["prompt_ids"]),
                             "response": _decode(trainer.tokenizer, row["response_ids"])} for row in records],
    })
    batch.meta_info["expert_action_sft"] = payload
    return {"expert_sft/enabled": 1.0, "expert_sft/coef": coef,
            "expert_sft/global_samples": len(records), "expert_sft/trajectories": trajectories,
            "expert_sft/library_trajectories": len(dataset.episodes), "expert_sft/library_decisions": len(dataset.rows),
            "expert_sft/supervised_tokens": sum(len(row["response_ids"]) for row in records),
            "expert_sft/action_tokens": sum(sum(row["action_mask"]) for row in records),
            "expert_sft/input_tokens": sum(len(row["prompt_ids"]) + len(row["response_ids"]) for row in records)}
