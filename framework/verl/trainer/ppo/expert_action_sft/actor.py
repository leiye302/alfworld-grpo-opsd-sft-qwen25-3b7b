"""Independent expert CE, attached to the last native optimizer mini-update.

Schema 3 averages supervised tokens over all trajectory draws and ranks.
Legacy Action-only and single-trajectory reductions retain their own contracts.
"""

import math
import time

import torch


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_payload(payload, options):
    """Validate the independent payload without consuming any random numbers."""
    trajectory = options.get("mode") == "complete_trajectory_reasoning_action"
    if trajectory:
        from verl.trainer.ppo.expert_action_sft.schedule import MODE, coefficient
        token_mean = options.get("loss_reduction", "sequence-mean") == "token-mean"
        dataset_format = options.get("dataset_format", "legacy_complete_decisions_v1")
        if dataset_format not in ("legacy_complete_decisions_v1", "mixed_expert_episode_v1"):
            raise ValueError("Unknown expert trajectory dataset format")
        mixed = dataset_format == "mixed_expert_episode_v1"
        if mixed and (not token_mean or not isinstance(payload, dict) or payload.get("dataset_format") != dataset_format):
            raise ValueError("Mixed complete trajectories require explicit format and global token-mean")
        schema = 4 if mixed else 3 if token_mean else 2
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != schema or payload.get("mode") != MODE:
            raise ValueError("Complete trajectory SFT schema/reduction/mode mismatch")
        iteration = _integer(payload.get("rollout_iteration"), "rollout_iteration", 1)
        coef = coefficient(options, iteration)
        if payload.get("coef") != coef or "global_samples" in options:
            raise ValueError("Trajectory SFT coefficient/counter or sample contract mismatch")
        count = _integer(payload.get("global_samples"), "global_samples", 0)
        if coef == 0:
            if count != 0 or payload.get("records") != []:
                raise ValueError("Zero SFT coefficient requires no expert computation")
            if token_mean and (payload.get("trajectory_count") != 0 or payload.get("trajectory_draws") != []
                               or payload.get("global_supervised_tokens") != 0 or payload.get("loss_reduction") != "token-mean"):
                raise ValueError("Zero coefficient must have an empty token-mean batch")
            return payload
        records = payload.get("records")
        if not isinstance(records, list) or count < 1 or len(records) != count or any(not isinstance(r, dict) for r in records):
            raise ValueError("Trajectory SFT requires a complete nonempty expert episode")
        if token_mean:
            trajectories = _integer(options.get("trajectories_per_round"), "trajectories_per_round", 1)
            draws = payload.get("trajectory_draws")
            if (payload.get("loss_reduction") != "token-mean" or payload.get("trajectory_count") != trajectories
                    or not isinstance(draws, list) or len(draws) != trajectories):
                raise ValueError("Global SFT trajectory count or token-mean contract mismatch")
            cursor, total_tokens = 0, 0
            for index, draw in enumerate(draws):
                if not isinstance(draw, dict) or draw.get("draw") != index or draw.get("start") != cursor:
                    raise ValueError("Trajectory draw boundary/order mismatch")
                decisions = _integer(draw.get("decisions"), "decisions", 1)
                eid = draw.get("episode_id")
                selected = records[cursor:cursor + decisions]
                if mixed:
                    total_decisions = _integer(draw.get("total_decisions"), "total_decisions", 1)
                    step_indices = draw.get("selected_step_indices")
                    if (not isinstance(step_indices, list) or len(step_indices) != decisions
                            or any(type(i) is not int or not 0 <= i < total_decisions for i in step_indices)
                            or step_indices != sorted(set(step_indices))):
                        raise ValueError("Mixed original step indices/total decisions mismatch")
                    expected_ids = [f"{eid}:{i}" for i in step_indices]
                    if (draw.get("selected_sample_ids") != expected_ids
                            or [r.get("source_step") for r in selected] != step_indices):
                        raise ValueError("Mixed selected sample identities/original steps mismatch")
                    source = draw.get("source_episode_id")
                    kind = draw.get("source_kind")
                    if (not isinstance(source, str) or not source or kind not in ("legacy_complete", "real_closedloop")
                            or eid != ("legacy:" if kind == "legacy_complete" else "closedloop:") + source
                            or any(r.get("episode_id") != eid or r.get("source_episode_id") != source
                                   or r.get("source_kind") != kind for r in selected)):
                        raise ValueError("Mixed complete source trajectory identity mismatch")
                    if kind == "legacy_complete" and step_indices != list(range(total_decisions)):
                        raise ValueError("Legacy complete draw cannot drop source decisions")
                else:
                    expected_ids = [f"{eid}:{i}" for i in range(decisions)]
                if (not isinstance(eid, str) or not eid or len(selected) != decisions
                        or [r.get("sample_id") for r in selected] != expected_ids):
                    raise ValueError("Complete trajectory identity/order is inconsistent")
                tokens = sum(len(row.get("response_ids", [])) for row in selected)
                if _integer(draw.get("supervised_tokens"), "draw supervised_tokens", 1) != tokens:
                    raise ValueError("Trajectory token count mismatch")
                total_tokens += tokens
                cursor += decisions
            if cursor != count or _integer(payload.get("global_supervised_tokens"), "global_supervised_tokens", 1) != total_tokens:
                raise ValueError("Global SFT token denominator or row coverage mismatch")
        else:
            eid = payload.get("episode_id")
            if not isinstance(eid, str) or not eid or [record.get("sample_id") for record in records] != [f"{eid}:{i}" for i in range(count)]:
                raise ValueError("Trajectory SFT identity/order is inconsistent")
        # Reuse native token/mask validation below, without imposing four rows.
        for record in records:
            if record.get("sft_mask") != [1] * len(record.get("response_ids", [])):
                raise ValueError("Full trajectory SFT must supervise every original response token")
    else:
        return _validate_action_payload(payload, options)
    _validate_records(records)
    return payload


def _validate_action_payload(payload, options):
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1):
        raise ValueError("Expert Action SFT requires payload schema_version=1")
    _integer(payload.get("rollout_iteration"), "rollout_iteration", 1)
    if options.get("global_samples") != 4 or payload.get("global_samples") != 4:
        raise ValueError("This experiment requires exactly four expert decisions per rollout")
    coef = float(options.get("coef", 0.01))
    payload_coef = float(payload.get("coef", float("nan")))
    if not math.isfinite(coef) or coef <= 0 or payload_coef != coef:
        raise ValueError("Expert Action SFT payload/config coefficients must agree and be positive")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("Expert Action SFT payload must contain exactly four records")
    _validate_records(records)
    return payload


def _validate_records(records):
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("sample_id"), str) or not record["sample_id"]:
            raise ValueError("Each expert decision needs a nonempty sample_id")
        for key in ("prompt_ids", "response_ids"):
            ids = record.get(key)
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"Expert {key} must be a nonempty token list")
            for token in ids:
                _integer(token, key)
        mask = record.get("action_mask")
        if (not isinstance(mask, list) or len(mask) != len(record["response_ids"])
                or any(type(value) is not int or value not in (0, 1) for value in mask)
                or not any(mask)):
            raise ValueError("Expert action_mask must align with responses and supervise at least one action token")
        _integer(record.get("pad_token_id"), "pad_token_id")


def prepare_auxiliary(actor, data):
    """Return a validated rank-local plan; disabled/zero coefficient is a no-op."""
    options = actor.config.get("expert_action_sft", {})
    if not options.get("enabled", False):
        return None
    coef = float(options.get("coef", 0.01))
    if not math.isfinite(coef) or coef < 0:
        raise ValueError("Expert Action SFT coefficient must be finite and nonnegative")
    if coef == 0:
        return None
    trajectory = options.get("mode") == "complete_trajectory_reasoning_action"
    payload = validate_payload(data.meta_info.get("expert_action_sft"), options) if trajectory else None
    if trajectory and payload["coef"] == 0:
        return None
    if (actor.config.use_dynamic_bsz or actor.use_ulysses_sp
            or int(actor.config.ppo_epochs) != 1 or "multi_modal_inputs" in data.non_tensor_batch):
        raise NotImplementedError("Expert Action SFT requires fixed microbatches, text, SP1, and PPO epochs=1")
    if not actor.config.get("use_sdar_loss", False) or actor.config.get("use_sdl_loss", False):
        raise ValueError("Expert Action SFT requires the unchanged independent GRPO+OPSD branch")
    if actor.config.get("use_bad_teacher_sft", False) or actor.config.get("use_bad_reset", False):
        raise ValueError("Expert Action SFT cannot be combined with prior Bad-only experiments")
    if not torch.distributed.is_initialized() or (torch.distributed.get_world_size() != 4 and not (trajectory and torch.distributed.get_world_size() == 8)):
        raise ValueError("Expert SFT requires four ranks, or eight ranks for complete-trajectory SFT")
    rank = torch.distributed.get_rank()
    payload = payload if trajectory else validate_payload(data.meta_info.get("expert_action_sft"), options)
    if data.batch.batch_size[0] <= 0:
        raise ValueError("Cannot attach auxiliary CE to an empty Student update")
    return {"payload": payload, "rank": rank, "coef": payload["coef"]}


def build_microbatch(records, rank, device):
    """Left-pad history, right-pad response, and rebuild causal positions."""
    record = records[rank]
    prompt_width = max(len(row["prompt_ids"]) for row in records)
    response_width = max(len(row["response_ids"]) for row in records)
    left = prompt_width - len(record["prompt_ids"])
    right = response_width - len(record["response_ids"])
    pad = record["pad_token_id"]
    responses = record["response_ids"] + [pad] * right
    ids = [pad] * left + record["prompt_ids"] + responses
    attention = [0] * left + [1] * (len(record["prompt_ids"]) + len(record["response_ids"])) + [0] * right
    attention_tensor = torch.tensor([attention], dtype=torch.long, device=device)
    batch = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": attention_tensor,
        "position_ids": (attention_tensor.cumsum(-1) - 1).clamp_min(0),
        "responses": torch.tensor([responses], dtype=torch.long, device=device),
    }
    mask = torch.tensor([record.get("sft_mask", record["action_mask"]) + [0] * right], dtype=torch.long, device=device)
    return batch, mask


def action_cross_entropy(log_probs, mask, reduction="mean"):
    """Masked target-token CE; global token normalization happens in the caller."""
    if reduction not in ("mean", "sum"):
        raise ValueError("Unsupported expert CE reduction")
    if log_probs.ndim != 2 or log_probs.shape != mask.shape or log_probs.shape[0] != 1:
        raise ValueError("Expected one expert decision per rank with aligned Action log-probabilities")
    if not bool(mask.sum() > 0):
        raise ValueError("An expert decision must contain supervised Action tokens")
    if log_probs.dtype in (torch.float16, torch.bfloat16):
        log_probs = log_probs.float()
    supervised = log_probs.masked_select(mask.bool())
    if not bool(torch.isfinite(supervised).all()):
        raise FloatingPointError("Nonfinite expert Action log-probability")
    return -supervised.sum() if reduction == "sum" else -supervised.mean()


def backward_auxiliary(actor, plan):
    """Add CE gradients to the existing final mini-batch; never step an optimizer."""
    if plan["payload"].get("mode") == "complete_trajectory_reasoning_action":
        return backward_trajectory(actor, plan)
    started = time.monotonic()
    device = next(actor.actor_module.parameters()).device
    batch, mask = build_microbatch(plan["payload"]["records"], plan["rank"], device)
    rng_devices = [torch.cuda.current_device()] if actor.device_name == "cuda" else []
    # Include backward so recomputed dropout from activation checkpointing also
    # leaves the native Student RNG stream untouched.
    with torch.random.fork_rng(devices=rng_devices):
        _, log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=False)
        loss = action_cross_entropy(log_probs, mask)
        (plan["coef"] * loss).backward()
    return {
        "expert_action_sft/loss": loss.detach().item(),
        "expert_action_sft/weighted_loss": plan["coef"] * loss.detach().item(),
        "expert_action_sft/coef": plan["coef"],
        "expert_action_sft/aux_batches": 1,
        "expert_action_sft/local_samples": 1,
        "expert_action_sft/global_samples": 4,
        "expert_action_sft/supervised_tokens_local": int(mask.sum().item()),
        "expert_action_sft/supervised_tokens_global": sum(
            sum(record["action_mask"]) for record in plan["payload"]["records"]),
        "expert_action_sft/rollout_iteration": plan["payload"]["rollout_iteration"],
        "expert_action_sft/seconds": time.monotonic() - started,
    }


def backward_trajectory(actor, plan):
    """Equal collective counts; FSDP's rank average yields the global CE.

    Token-mean: each real decision contributes W/T times its token CE SUM,
    where T covers all selected trajectories, never the local rank/microbatch.
    Legacy sequence-mean uses W/N times each sequence mean. Padding contributes
    zero while preserving collective counts. No division by accumulation rounds.
    """
    started = time.monotonic()
    records = plan["payload"]["records"]
    count, world = len(records), torch.distributed.get_world_size()
    token_mean = plan["payload"].get("loss_reduction") == "token-mean"
    denominator = plan["payload"]["global_supervised_tokens"] if token_mean else count
    rank = plan["rank"]
    rounds = (count + world - 1) // world
    device = next(actor.actor_module.parameters()).device
    rng_devices = [torch.cuda.current_device()] if actor.device_name == "cuda" else []
    local_mean, local_count, local_tokens = 0.0, 0, 0
    with torch.random.fork_rng(devices=rng_devices):
        for round_index in range(rounds):
            start = round_index * world
            group = records[start:start + world]
            real = rank < len(group)
            assigned = rank if real else 0
            batch, mask = build_microbatch(group, assigned, device)
            _, log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=False)
            loss = action_cross_entropy(log_probs, mask, reduction="sum" if token_mean else "mean")
            scale = world / denominator if real else 0.0
            (plan["coef"] * scale * loss).backward()
            local_mean += scale * loss.detach().item()
            local_count += int(real)
            local_tokens += int(mask.sum().item()) if real else 0
    return {
        # Native metric reduction averages ranks; this becomes the global CE.
        "expert_action_sft/loss": local_mean,
        "expert_action_sft/weighted_loss": plan["coef"] * local_mean,
        "expert_action_sft/coef": plan["coef"],
        "expert_action_sft/aux_batches": rounds,
        "expert_action_sft/local_samples": local_count,
        "expert_action_sft/global_samples": count,
        "expert_action_sft/trajectories": plan["payload"].get("trajectory_count", 1),
        "expert_action_sft/global_token_mean": float(token_mean),
        "expert_action_sft/supervised_tokens_local": local_tokens,
        "expert_action_sft/supervised_tokens_global": sum(len(row["response_ids"]) for row in records),
        "expert_action_sft/rollout_iteration": plan["payload"]["rollout_iteration"],
        "expert_action_sft/seconds": time.monotonic() - started,
    }
