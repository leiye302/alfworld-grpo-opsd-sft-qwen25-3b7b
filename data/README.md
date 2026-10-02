# ALFWorld expert data and fixed validation

`expert/episodes.jsonl` contains 409 complete source episodes: 353 `legacy_complete` expert-action episodes with offline reasoning annotations and 56 `real_closedloop` accepted Teacher episodes. All source games are ALFWorld **train** games. Game paths are relative to the released ALFWorld asset root.

Each JSONL line has a unique `episode_id`, `source_episode_id`, `kind` and schema. A legacy episode contains contiguous decision `rows` and its complete-episode index. A real episode contains its actual `trajectory.steps` and corresponding `quality.steps`; the latter specify `supervise`/`sft_step_mask` per original step. Student takeover prefixes and rejected Teacher steps stay in the history but do not receive SFT. Original step indices are preserved.

For each approved decision, the loader uses its real/native history prompt and the original reasoning/action response. It applies the selected model's chat template once and supervises the complete generated response, protocol tags and first EOS. Template-provided prompt tokens, observations and padding are excluded. Whole-trajectory sampling does not concatenate different examples into one shared causal attention sequence.

Legacy reasoning was annotated by Qwen2.5-72B with the designated expert action supplied. This is offline SFT data, not an action-blind reasoning evaluation. Closed-loop examples are actual executed Teacher responses, including straightforward completion, search and recovery. Masked error/prefix steps are context rather than imitation targets. Format/state/mask checks do not prove every reasoning sentence is optimal.

`expert/manifest.json` lists episode identities, their selected source-step indices and counts. It contains ordinary reproducible round seeds for the default SFT seed. These retain the pre-cleanup default sampling stream while removing any dependency on file bytes, author directories or hashes. Editing metadata or JSON whitespace does not invalidate the dataset. Substantive edits must keep the episode index, history chain and supervision flags consistent.

`fixed_validation/manifest.json` contains the same fixed 128 **valid_seen** task IDs, environment seeds, initial observations and evaluation settings. Its `gamefile` entries are relative; the runtime resolves them under the user's own ALFWorld asset root. Task/turn generation seeds are deterministic. No author directory or old game/manifest checksum is required. Keep this task list and evaluation protocol unchanged when comparing method and baseline.

The offline data has no API credentials and does not require an Analyzer or a 72B model at training time. Environment game data comes from ALFWorld/ALFRED; respect their upstream licenses and source attribution. Models used for annotation and training have their own licenses. Treat dataset version/name, task IDs, seeds and effective training configuration as the reproducibility record.
