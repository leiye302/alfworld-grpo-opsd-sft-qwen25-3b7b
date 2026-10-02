#!/usr/bin/env python3
"""CPU-only regression checks; no model downloads, account or GPU needed."""
import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'runtime'))
import handoff
import gpu_uuid_compat
import collect_results


def expert_modules():
    # Dataset structure is pure Python; avoid importing the GPU training stack.
    directory = ROOT/'framework/verl/trainer/ppo/expert_action_sft'
    package = ModuleType('public_expert')
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    result = {}
    for name in ('data', 'mixed_dataset', 'schedule'):
        spec = importlib.util.spec_from_file_location('public_expert.'+name, directory/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[name] = module
    return result


class PublicProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expert = expert_modules()
        cls.episodes = [json.loads(line) for line in
                        (ROOT/'data/expert/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        cls.index = json.loads((ROOT/'data/expert/manifest.json').read_text(encoding='utf-8'))

    def test_all_episode_identities_histories_actions_and_supervision(self):
        mixed = self.expert['mixed_dataset']
        entries = {row['episode_id']: row for row in self.index['episodes']}
        self.assertEqual(len(entries), len(self.episodes))
        self.assertEqual(len(entries), 409)
        counts = {'legacy_complete': 0, 'real_closedloop': 0}
        for episode in self.episodes:
            kind = episode['kind']
            rows, selected = (mixed._legacy_rows(episode) if kind == 'legacy_complete'
                              else mixed._real_rows(episode))
            info = entries[episode['episode_id']]
            self.assertEqual(info['source_episode_id'], episode['source_episode_id'])
            self.assertEqual(info['source_decisions'], len(rows))
            self.assertEqual(info['selected_step_indices'], selected)
            self.assertEqual(info['supervised_decisions'], len(selected))
            counts[kind] += 1
        self.assertEqual(counts, {'legacy_complete': 353, 'real_closedloop': 56})

    def test_dataset_and_validation_are_portable_without_file_hash_fields(self):
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(any('sha256' in key.lower() for key in value))
                for child in value.values(): check(child)
            elif isinstance(value, list):
                for child in value: check(child)
        check(self.episodes); check(self.index)
        fixed = json.loads((ROOT/'data/fixed_validation/manifest.json').read_text(encoding='utf-8'))
        check(fixed)
        self.assertEqual(fixed['validation_id'], handoff.VALIDATION_ID)
        self.assertEqual(len(fixed['tasks']), 128)
        self.assertEqual(len({row['task_id'] for row in fixed['tasks']}), 128)
        for row in fixed['tasks']:
            relative = PurePosixPath(row['gamefile'])
            self.assertEqual(relative.parts[:2], ('json_2.1.1', 'valid_seen'))
            self.assertFalse(relative.is_absolute())
            self.assertNotIn('..', relative.parts)
            self.assertEqual(row['gamefile'], row['task_id'])

    def test_sampling_ignores_file_paths_and_does_not_consume_global_rng(self):
        cls = self.expert['mixed_dataset'].MixedExpertTrajectoryDataset
        dataset = object.__new__(cls)
        dataset.sampling_seed = self.index['sampling_seed']
        dataset.round_seeds = self.index['round_seeds']
        dataset.episode_ids = sorted(row['episode_id'] for row in self.index['episodes'])
        dataset.episode_info = {row['episode_id']: row for row in self.index['episodes']}
        dataset.encoded, dataset.episodes = [], {}
        for eid in dataset.episode_ids:
            indices = []
            for step in dataset.episode_info[eid]['selected_step_indices']:
                indices.append(len(dataset.encoded))
                dataset.encoded.append({'sample_id': f'{eid}:{step}', 'sft_mask': [1]})
            dataset.episodes[eid] = indices
        before = random.getstate()
        self.assertEqual(len(dataset.round_seeds), 49)
        for iteration in (1, 25, 49, 60):
            first = dataset.sample_trajectories(dataset.sampling_seed, iteration, 16)
            dataset.dataset_path = '/another/project/expert.jsonl'
            self.assertEqual(first, dataset.sample_trajectories(dataset.sampling_seed, iteration, 16))
            self.assertEqual(len(first[2]), 16)
        self.assertEqual(random.getstate(), before)

    def test_bad_history_or_student_prefix_supervision_is_rejected(self):
        mixed = self.expert['mixed_dataset']
        legacy = copy.deepcopy(next(row for row in self.episodes if row['kind'] == 'legacy_complete'))
        legacy['rows'][1]['expert_history'][0]['action'] = 'unrelated command'
        with self.assertRaises(ValueError): mixed._legacy_rows(legacy)
        takeover = copy.deepcopy(next(row for row in self.episodes if row['kind'] == 'real_closedloop' and
                                     row['episode']['trajectory'].get('takeover_at', 0) > 0))
        takeover['episode']['quality']['steps'][0].update(supervise=True, sft_step_mask=1)
        with self.assertRaises(ValueError): mixed._real_rows(takeover)

    def test_native_protocol_parsing_keeps_the_executed_action(self):
        parse = self.expert['mixed_dataset'].parse_native_response
        for text in ('<think>Search the room.</think><action>go to desk 1</action>',
                     '<THINK>Search the room.</THINK>\n<ACTION>go to desk 1</ACTION>'):
            self.assertEqual(parse(text)[1], 'go to desk 1')
        for text in ('<think></think><action>go to desk 1</action>',
                     '<think>Search</think><act>go to desk 1</act>',
                     '<think>Search</think><action>go to desk 1\nlook</action>'):
            with self.assertRaises(ValueError): parse(text)

    def test_cosine_schedule_and_default_3b_only_configuration(self):
        args = handoff.arguments(Path('/data/project'), Path('/data/project/runs/3b_sft'), '3b', 'sft')
        options = {arg.lstrip('+').split('=', 1)[0]: arg.split('=', 1)[1] for arg in args}
        self.assertFalse(any('sha256' in key for key in options))
        expected = {'trainer.total_epochs': '150', 'trainer.n_gpus_per_node': '8',
                    'trainer.save_freq': '50', 'trainer.test_freq': '10',
                    'data.train_batch_size': '16', 'env.rollout.n': '8',
                    'actor_rollout_ref.actor.ppo_mini_batch_size': '256',
                    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': '32',
                    'algorithm.sdar.sdar_coef': '0.01', 'algorithm.sdar.gate_beta': '0.0'}
        for key, value in expected.items(): self.assertEqual(options[key], value, key)
        self.assertEqual(handoff.DEFAULT_RUNS, (('3b', 'sft'),))
        config = {'coef': .1, 'decay_end_iteration': 50, 'decay_schedule': 'cosine'}
        coefficient = self.expert['schedule'].coefficient
        self.assertAlmostEqual(coefficient(config, 1), .1)
        self.assertEqual(coefficient(config, 50), 0.)
        self.assertEqual(coefficient(config, 150), 0.)
        self.assertGreater(coefficient(config, 25), coefficient(config, 49))

    def test_prepare_downloads_only_the_selected_model_without_hash_gate(self):
        selected = []
        hub = ModuleType('huggingface_hub')
        hub.snapshot_download = lambda **kwargs: selected.append(kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'huggingface_hub': hub}), \
             patch.dict(os.environ, {}, clear=False), patch.object(handoff, 'download_asset') as download, \
             patch.object(handoff, 'extract_verified'), patch.object(handoff, 'digest', side_effect=AssertionError('Unexpected hash gate')), \
             contextlib.redirect_stdout(io.StringIO()):
            handoff.prepare(Path(temp))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['repo_id'], 'Qwen/Qwen2.5-3B-Instruct')
        self.assertFalse(download.call_args.args[2])

    def test_asset_extraction_rejects_escaping_paths_and_links(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for name in ('../outside', '/outside', 'C:/outside', 'dir\\outside'):
                with self.assertRaises(RuntimeError): handoff.asset_member(base/'data', name)
            archive = base/'assets.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                info = tarfile.TarInfo('../outside'); info.size = 1
                stream.addfile(info, io.BytesIO(b'x'))
            with self.assertRaises(RuntimeError): handoff.extract_verified(archive, base/'data')
            self.assertFalse((base/'outside').exists())
            with tarfile.open(archive, 'w:gz') as stream:
                info = tarfile.TarInfo('link'); info.type = tarfile.SYMTYPE; info.linkname = '/outside'
                stream.addfile(info)
            with self.assertRaises(RuntimeError): handoff.extract_verified(archive, base/'data')

    def test_missing_resume_state_never_starts_training(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(handoff.subprocess, 'run') as launch:
            with self.assertRaisesRegex(RuntimeError, 'No complete checkpoint'):
                handoff.run_one(Path(temp), '3b', 'sft', True)
            launch.assert_not_called()

    def test_result_export_rejects_mismatched_validation_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'runs/3b_sft/fixed_validation/results'
            directory.mkdir(parents=True)
            (directory/'iteration_000010.json').write_text(json.dumps(
                {'iteration': 10, 'validation_id': 'another-split', 'tasks': 128,
                 'successes': 64, 'success_rate': .5}), encoding='utf-8')
            with patch.object(sys, 'argv', ['collect_results', '--work', temp]):
                with self.assertRaisesRegex(RuntimeError, 'fixed 128-task protocol'):
                    collect_results.main()

    def test_gpu_uuid_conversion_supports_all_eight_devices_in_order(self):
        class Platform:
            device_control_env_var = 'CUDA_VISIBLE_DEVICES'
            @classmethod
            def device_id_to_physical_device_id(cls, index): return index
        nvml = ModuleType('pynvml')
        nvml.nvmlInit = nvml.nvmlShutdown = lambda: None
        nvml.nvmlDeviceGetHandleByUUID = lambda uuid: int(uuid.split('-')[1])
        nvml.nvmlDeviceGetIndex = lambda handle: handle
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'pynvml': nvml}), \
             patch.dict(os.environ, {'SDAR_RUN_ROOT': temp, 'SDAR_ALLOWED_GPU_COUNT': '8',
                                     'CUDA_VISIBLE_DEVICES': 'GPU-7,GPU-6,GPU-5,GPU-4,GPU-3,GPU-2,GPU-1,GPU-0'}):
            (Path(temp)/'logs').mkdir()
            gpu_uuid_compat.physical_index.cache_clear()
            gpu_uuid_compat.install(SimpleNamespace(Platform=Platform))
            self.assertEqual([Platform.device_id_to_physical_device_id(i) for i in range(8)], list(reversed(range(8))))
            with self.assertRaises(AssertionError): gpu_uuid_compat.physical_index('GPU-8')
        gpu_uuid_compat.physical_index.cache_clear()

    def test_current_project_has_no_author_machine_paths_or_private_keys(self):
        forbidden = re.compile(rb'/mnt/[^/\s]+/test/VLA_test|connect\.[a-z0-9.-]+\.seetacloud\.com|[a-z0-9-]+\.cloud\.infini-ai\.com')
        private_key = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
        skip = {'.git', '.venv', 'work', '__pycache__', '.bootstrap_cache', '.bootstrap_tmp',
                'outputs', 'multirun', 'logs', 'cache'}
        for path in ROOT.rglob('*'):
            if not path.is_file() or any(part in skip for part in path.relative_to(ROOT).parts): continue
            content = path.read_bytes()
            self.assertIsNone(forbidden.search(content), path.relative_to(ROOT).as_posix())
            self.assertIsNone(private_key.search(content), path.relative_to(ROOT).as_posix())

    def test_public_python_and_shell_sources_parse_and_use_lf(self):
        for directory in ('scripts', 'runtime'):
            for path in (ROOT/directory).glob('*.py'):
                ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
        for path in (ROOT/'scripts').glob('*.sh'):
            self.assertNotIn(b'\r', path.read_bytes(), path.name)
        launch = (ROOT/'scripts/launch.sh').read_text(encoding='utf-8')
        worker = (ROOT/'scripts/run_3b.sh').read_text(encoding='utf-8')
        self.assertIn('tmux new-session -d', launch)
        self.assertIn('scripts/run_3b.sh', launch)
        self.assertIn('scripts/bootstrap.sh', worker)
        self.assertIn('prepare --work', worker)
        self.assertIn('--size 3b --method sft', worker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    options = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PublicProjectChecks))
    report = {'passed': result.wasSuccessful(), 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'expert_trajectories': 409, 'fixed_validation_tasks': 128,
              'gpu_training_tested': False, 'real_tokenizer_tested_here': False}
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__': main()
