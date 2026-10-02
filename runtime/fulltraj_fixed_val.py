"""Fixed 128-task validation only; training paths call their native methods."""
from collections import defaultdict
from copy import copy, deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

ROOT = Path(os.environ['SDAR_RUN_ROOT']) / 'fixed_validation'
MANIFEST_PATH = ROOT / 'manifest.json'

def write_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT.resolve())
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()

def manifest():
    data = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    assert data['validation_id'] == 'alfworld-valid-seen-128-v1'
    assert len(data['tasks']) == 128
    assert len({row['task_id'] for row in data['tasks']}) == 128
    root = Path(os.environ['ALFWORLD_DATA'])
    for task in data['tasks']:
        relative = PurePosixPath(task['gamefile'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[:2] == ('json_2.1.1', 'valid_seen')
        assert relative.as_posix() == task['task_id']
        task['gamefile'] = str(root.joinpath(*relative.parts))
    return data

def request_seed(task_id, turn):
    value = f'alfworld-fixed-val128-v1|20260926|{task_id}|{int(turn)}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'big') % (2**31 - 1)

def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value

def install_envs(module):
    cls = module.AlfworldWorker
    native_init, native_reset = cls.__init__, cls.reset

    def initialize(self, config, seed, base_env):
        self._fixed_task = None
        if base_env.train_eval == 'eval_in_distribution':
            index = int(seed) - 1000
            data = manifest()
            assert 0 <= index < 128
            task = data['tasks'][index]
            assert task['gamefile'] in base_env.game_files
            base_env = copy(base_env)
            base_env.game_files, base_env.num_games = [task['gamefile']], 1
            self._fixed_task = task
            seed = task['environment_seed']
        native_init(self, config, seed, base_env)

    def reset(self):
        if self._fixed_task is not None:
            self.env.seed(self._fixed_task['environment_seed'])
        obs, infos = native_reset(self)
        if self._fixed_task is not None:
            assert infos['extra.gamefile'] == [self._fixed_task['gamefile']]
        return obs, infos

    cls.__init__, cls.reset = initialize, reset
    # Mark manager instances without altering any native constructor argument.
    env_class = module.AlfworldEnvs
    env_init = env_class.__init__
    import inspect
    signature = inspect.signature(env_init)
    def group_initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        self._fulltraj_is_train = bool(bound.arguments['is_train'])
        return env_init(self, *args, **kwargs)
    env_class.__init__ = group_initialize


def install_manager(module):
    cls = module.AlfWorldEnvironmentManager
    native_reset, native_step = cls.reset, cls.step

    def reset(self, kwargs):
        if self.envs._fulltraj_is_train:
            return native_reset(self, kwargs)
        obs, infos = native_reset(self, kwargs)
        tasks = manifest()['tasks']
        assert self.gamefile == [row['gamefile'] for row in tasks]
        self._fixed_records = [[] for _ in tasks]
        self._fixed_initial = []
        for i, task in enumerate(tasks):
            state = {'observation': obs['anchor'][i],
                     'admissible_commands': self.envs.get_admissible_commands[i]}
            assert state['observation'] == task['initial_observation'], task['task_id']
            self._fixed_initial.append(state)
        return obs, infos

    def step(self, text_actions):
        if self.envs._fulltraj_is_train:
            return native_step(self, text_actions)
        before = list(self.pre_text_obs)
        result = native_step(self, text_actions)
        obs, rewards, dones, infos = result
        for i, response in enumerate(text_actions):
            self._fixed_records[i].append({
                'observation_before': before[i], 'response': response,
                'observation_after': obs['anchor'][i],
                'reward': float(rewards[i]), 'done': bool(dones[i]),
                'won': bool(infos[i]['won']),
                'format_valid': bool(infos[i]['is_action_valid']),
                'gamefile': infos[i]['extra.gamefile'],
            })
        write_json(ROOT / 'reports/progress.json', {
            'phase': 'evaluating', 'checkpoint': getattr(self, '_fixed_label', None),
            'environment_turn': len(self._fixed_records[0]),
            'time': time.time(),
        })
        return result

    cls.reset, cls.step = reset, step

def install_collector(module):
    cls = module.TrajectoryCollector
    native_preprocess = cls.preprocess_batch
    native_loop = cls.vanilla_multi_turn_loop

    def preprocess(self, gen_batch, obs):
        import numpy as np
        if gen_batch.meta_info.get('validate') is not True:
            return native_preprocess(self, gen_batch, obs)
        result = native_preprocess(self, gen_batch, obs)
        tasks = manifest()['tasks']
        assert list(obs['gamefile']) == [t['gamefile'] for t in tasks]
        result.non_tensor_batch['tools_kwargs'] = np.array([
            {'fixed_eval_task_id': t['task_id'], 'fixed_eval_turn': self._fixed_turn,
             'fixed_eval_seed': request_seed(t['task_id'], self._fixed_turn)}
            for t in tasks
        ], dtype=object)
        self._fixed_turn += 1
        return result

    def loop(self, gen_batch, actor_rollout_wg, envs):
        if gen_batch.meta_info.get('validate') is not True:
            return native_loop(self, gen_batch, actor_rollout_wg, envs)
        self._fixed_turn = 0
        result = native_loop(self, gen_batch, actor_rollout_wg, envs)
        batches, rewards, lengths, successes, trajectory_ids, tools = result
        tasks = manifest()['tasks']
        assert len(batches) == len(tasks) == 128
        records = []
        for i, task in enumerate(tasks):
            steps = []
            for turn, row in enumerate(batches[i]):
                if not bool(row['active_masks']):
                    continue
                item = dict(envs._fixed_records[i][turn])
                assert item['gamefile'] == task['gamefile']
                metadata = row['tools_kwargs']
                assert metadata['fixed_eval_task_id'] == task['task_id']
                assert metadata['fixed_eval_turn'] == turn
                assert metadata['fixed_eval_seed'] == request_seed(task['task_id'], turn)
                prompts = plain(row['prompts'])
                responses = plain(row['responses'])
                mask = plain(row['attention_mask'])
                item.update(turn=turn, sampling_seed=metadata['fixed_eval_seed'],
                            prompt_ids=[x for x, m in zip(prompts, mask[:len(prompts)]) if m],
                            response_ids=[x for x, m in zip(responses, mask[len(prompts):]) if m])
                steps.append(item)
            assert len(steps) == int(lengths[i]) and 1 <= len(steps) <= 50
            won = steps[-1]['won']
            assert sum(s['reward'] for s in steps) == float(rewards[i])
            assert bool(float(rewards[i]) == 10.) == won
            records.append({**task, 'initial_state': envs._fixed_initial[i],
                            'success': won, 'episode_reward': float(rewards[i]),
                            'length': len(steps), 'steps': steps})
        assert sum(x['success'] for x in records) / 128 == float(module.np.mean(successes['success_rate']))
        self._fixed_records = records
        return result

    cls.preprocess_batch, cls.vanilla_multi_turn_loop = preprocess, loop

def install_vllm(module):
    cls = module.vLLMRollout
    native = cls.generate_sequences

    def generate(self, prompts, **kwargs):
        if prompts.meta_info.get('validate') is not True:
            return native(self, prompts, **kwargs)
        metadata = list(prompts.non_tensor_batch['tools_kwargs'])
        engine_generate = self.inference_engine.generate

        def seeded_generate(*args, **engine_kwargs):
            assert not args
            inputs = engine_kwargs['prompts']
            assert len(inputs) == len(metadata)
            parameters = engine_kwargs['sampling_params']
            assert parameters.temperature == 0.4 and parameters.n == 1
            assert parameters.max_tokens == 512
            per_request = []
            for identity in metadata:
                expected = request_seed(identity['fixed_eval_task_id'], identity['fixed_eval_turn'])
                assert expected == identity['fixed_eval_seed']
                item = deepcopy(parameters)
                item.seed = expected
                per_request.append(item)
            engine_kwargs['sampling_params'] = per_request
            outputs = engine_generate(**engine_kwargs)
            import torch
            rank = torch.distributed.get_rank()
            # Record actual engine requests; ranks in a TP pair intentionally repeat them.
            with (ROOT / 'logs' / f'sampling_rank{rank}.jsonl').open('a') as handle:
                handle.write(json.dumps({'time': time.time(), 'requests': metadata,
                    'prompt_sha256': [digest(p['prompt_token_ids']) for p in inputs],
                    'output_sha256': [digest(list(o.outputs[0].token_ids)) for o in outputs]}) + '\n')
            return outputs

        self.inference_engine.generate = seeded_generate
        try:
            return native(self, prompts, **kwargs)
        finally:
            self.inference_engine.generate = engine_generate

    cls.generate_sequences = generate


def install_validation(module):
    cls = module.RayPPOTrainer
    native = cls._validate

    def validate(self):
        import random
        import numpy as np
        import torch
        # Validation identities/sampling must not perturb host training RNG.
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        started = time.time()
        iteration = int(self.global_steps)
        fixed = manifest()
        assert self.config.data.val_batch_size == 128 and self.config.env.max_steps == 50
        assert self.config.env.history_length == 2
        self.val_envs._fixed_label = f'iteration_{iteration:06d}'
        try:
            metrics = native(self)
            records = self.traj_collector._fixed_records
            successes = sum(row['success'] for row in records)
            assert len(records) == 128 and metrics['val/success_rate'] == successes / 128
            out = ROOT / 'results' / f'iteration_{iteration:06d}'
            with gzip.open(str(out) + '.trajectories.jsonl.gz', 'wt', encoding='utf-8') as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            write_json(str(out) + '.json', {
                'iteration': iteration, 'validation_id': fixed['validation_id'],
                'tasks': 128, 'successes': successes, 'success_rate': successes / 128,
                'native_metrics': plain(metrics), 'seconds': time.time() - started,
                'per_task': [{key: row[key] for key in ('task_id', 'task_type',
                    'environment_seed', 'success', 'length', 'episode_reward')} for row in records],
            })
            print('[FIXED_VALIDATION_COMPLETE]', iteration, successes, '/128', flush=True)
            return metrics
        finally:
            random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    cls._validate = validate
