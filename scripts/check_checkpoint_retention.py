#!/usr/bin/env python3
"""CPU-only failure and deletion-scope checks for the handoff checkpoint policy."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime'))
sys.path.insert(0, str(ROOT/'scripts'))
import checkpoint_retention as retention
import storage_guard
import handoff


def write_state(path, data=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', json.dumps(data if data is not None else {'fixture': True}))
        archive.writestr('archive/version', '3\n')
        archive.writestr('archive/data/0', b'fixture tensor bytes')


def make_checkpoint(run, iteration, marker=True):
    directory = run/'checkpoints'/f'global_step_{iteration}'
    actor = directory/'actor'
    actor.mkdir(parents=True, exist_ok=True)
    for kind in ('model', 'optim', 'extra_state'):
        for rank in range(8):
            write_state(actor/f'{kind}_world_size_8_rank_{rank}.pt')
    write_state(directory/'data.pt')
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (actor/name).write_text('{}')
    if marker:
        (run/'checkpoints/latest_checkpointed_iteration.txt').write_text(str(iteration))
    return directory


def native_save_method():
    """Execute the actual frozen trainer save method with CPU file-writer stubs."""
    tree = ast.parse((ROOT/'framework/verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8'))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RayPPOTrainer')
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == '_save_checkpoint')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {'os': os, 'torch': SimpleNamespace(save=lambda data, path: write_state(Path(path), data))}
    exec(compile(module, '<actual-native-save>', 'exec'), namespace)
    return namespace['_save_checkpoint']


class TrainerOptions(dict):
    __getattr__ = dict.__getitem__


class RetentionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name).resolve()/'runs/3b_sft'
        (self.run/'logs').mkdir(parents=True)
        (self.run/'checkpoints').mkdir()
        self.rng = random.getstate()

    def tearDown(self):
        self.assertEqual(random.getstate(), self.rng, 'Retention changed the Python random stream')
        self.temporary.cleanup()

    def test_successive_saves_only_retire_named_previous_checkpoint(self):
        other = self.run.parent/'7b_baseline'
        (other/'checkpoints/global_step_50').mkdir(parents=True)
        untouched = other/'checkpoints/global_step_50/evidence.txt'
        untouched.write_bytes(b'other experiment')
        sampling = self.run/'checkpoints/expert_action_sft_sampling'
        sampling.mkdir()
        (sampling/'receipts.jsonl').write_bytes(b'keep sampling evidence')
        for step in (50,100,150):
            make_checkpoint(self.run,step)
            retention.after_native_save(self.run,step)
            self.assertTrue((self.run/'checkpoints'/f'global_step_{step}').is_dir())
        self.assertFalse((self.run/'checkpoints/global_step_50').exists())
        self.assertFalse((self.run/'checkpoints/global_step_100').exists())
        self.assertEqual(untouched.read_bytes(),b'other experiment')
        self.assertEqual((sampling/'receipts.jsonl').read_bytes(),b'keep sampling evidence')

    def test_missing_rank_preserves_old_and_restores_resume_marker(self):
        old = make_checkpoint(self.run,50)
        new = make_checkpoint(self.run,100)
        (new/'actor/optim_world_size_8_rank_7.pt').unlink()
        with self.assertRaises(RuntimeError):retention.after_native_save(self.run,100)
        self.assertTrue(old.is_dir())
        self.assertEqual((self.run/'checkpoints/latest_checkpointed_iteration.txt').read_text(),'50')

    def test_nonempty_truncated_shard_preserves_previous(self):
        old = make_checkpoint(self.run,50)
        new = make_checkpoint(self.run,100)
        shard = new/'actor/model_world_size_8_rank_3.pt'
        shard.write_bytes(shard.read_bytes()[:-40])
        self.assertGreater(shard.stat().st_size,0)
        with self.assertRaises(RuntimeError):retention.after_native_save(self.run,100)
        self.assertTrue(old.is_dir())
        self.assertEqual((self.run/'checkpoints/latest_checkpointed_iteration.txt').read_text(),'50')

    def test_missing_data_state_and_wrong_marker_preserve_previous(self):
        old = make_checkpoint(self.run,50)
        new = make_checkpoint(self.run,100)
        (new/'data.pt').unlink()
        with self.assertRaises(RuntimeError):retention.after_native_save(self.run,100)
        self.assertTrue(old.is_dir())
        make_checkpoint(self.run,100)
        (self.run/'checkpoints/latest_checkpointed_iteration.txt').write_text('75')
        with self.assertRaises(RuntimeError):retention.after_native_save(self.run,100)
        self.assertTrue(old.is_dir())
        self.assertEqual((self.run/'checkpoints/latest_checkpointed_iteration.txt').read_text(),'50')

    def test_refuses_unapproved_run_and_iteration(self):
        make_checkpoint(self.run,50)
        with self.assertRaises(RuntimeError):retention.after_native_save(self.run,25)
        other = self.run.parent/'legacy_method'
        other.mkdir()
        with self.assertRaises(RuntimeError):retention.after_native_save(other,50)
        self.assertTrue((self.run/'checkpoints/global_step_50').is_dir())

    def test_refuses_redirected_previous_directory(self):
        outside = Path(self.temporary.name)/'outside'
        outside.mkdir()
        (outside/'keep.txt').write_text('keep')
        make_checkpoint(self.run,100)
        old = self.run/'checkpoints/global_step_50'
        try:old.symlink_to(outside,target_is_directory=True)
        except OSError as error:self.skipTest('Symlinks unavailable on this host: '+str(error))
        with self.assertRaises(RuntimeError):retention.after_native_save(self.run,100)
        self.assertEqual((outside/'keep.txt').read_text(),'keep')
        self.assertTrue(old.is_symlink())

    def test_native_hook_verifies_after_save_and_keeps_backup_on_rpc_failure(self):
        run = self.run
        class FakeTrainer:
            _save_checkpoint = native_save_method()
            def _dump_generations(self,*args,**kwargs):return None
        self.assertEqual(FakeTrainer._save_checkpoint.__name__,'_save_checkpoint')
        storage_guard.install_trainer_hooks(SimpleNamespace(RayPPOTrainer=FakeTrainer))
        trainer = FakeTrainer()
        trainer.config = SimpleNamespace(trainer=TrainerOptions(default_local_dir=str(run/'checkpoints'),
            default_hdfs_dir=None,max_actor_ckpt_to_keep=None,max_critic_ckpt_to_keep=None))
        trainer.use_critic = False
        trainer.train_dataloader = SimpleNamespace(state_dict=lambda:{'position':100})
        calls = []
        def save_actor(path,remote,iteration,max_ckpt_to_keep=None):
            calls.append(iteration)
            self.assertIsNone(max_ckpt_to_keep)
            if iteration>50:self.assertTrue((run/'checkpoints'/f'global_step_{iteration-50}').is_dir())
            make_checkpoint(run,iteration,marker=False)
        trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=save_actor)
        reserve = SimpleNamespace(maintain=lambda **kwargs:None)
        with patch.dict(os.environ,{'SDAR_RUN_ROOT':str(run),'SDAR_CHECKPOINT_RETENTION_LAST_ONLY':'1'}), \
             patch.object(storage_guard,'for_run',return_value=reserve):
            for step in (50,100):
                trainer.global_steps=step;trainer._save_checkpoint()
            self.assertEqual(calls,[50,100])
            self.assertFalse((run/'checkpoints/global_step_50').exists())
            trainer.global_steps=150
            def broken_save(*args,**kwargs):
                (run/'checkpoints/latest_checkpointed_iteration.txt').write_text('150')
                raise OSError('simulated failed distributed save')
            trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=broken_save)
            with self.assertRaises(OSError):trainer._save_checkpoint()
            self.assertTrue((run/'checkpoints/global_step_100').is_dir())
            self.assertEqual((run/'checkpoints/latest_checkpointed_iteration.txt').read_text(),'100')

    def test_four_profiles_only_change_save_frequency_from_recorded_recipe(self):
        original=json.loads((ROOT/'configs/recorded_1p5b_launch_args.json').read_text())
        previous={arg.lstrip('+').split('=',1)[0]:arg.split('=',1)[1] for arg in original}
        excluded={'data.train_files','data.val_files','actor_rollout_ref.model.path','trainer.experiment_name',
          'trainer.n_gpus_per_node','ray_init.num_gpus','trainer.total_epochs','trainer.save_freq',
          'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir','ray_init._temp_dir',
          'ray_init._plasma_directory','trainer.resume_mode','actor_rollout_ref.actor.expert_action_sft.enabled',
          'actor_rollout_ref.actor.expert_action_sft.dataset_path',
          'actor_rollout_ref.actor.expert_action_sft.trajectory_manifest_path'}
        for size in ('3b','7b'):
            for method in ('baseline','sft'):
                args=handoff.arguments(self.run.parent.parent,self.run,size,method)
                actual={arg.lstrip('+').split('=',1)[0]:arg.split('=',1)[1] for arg in args}
                self.assertEqual(actual['trainer.save_freq'],'50')
                self.assertEqual(actual['trainer.test_freq'],'10')
                for key,value in previous.items():
                    if key not in excluded:self.assertEqual(actual[key],value,key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    options=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RetentionChecks))
    if options.report:
        report={'passed':result.wasSuccessful(),'tests':result.testsRun,'skipped':len(result.skipped),
                'failure_modes_checked':['missing rank','nonempty truncated shard','missing data state',
                  'wrong marker','RPC failure','unapproved path/iteration','cross-run preservation'],
                'actual_native_save_method_executed':True,'gpu_used':False,
                'checkpoint_policy':retention.POLICY,'real_3b_7b_checkpoint_tensor_reload_tested':False,
                'files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                  ('runtime/checkpoint_retention.py','runtime/storage_guard.py','scripts/handoff.py','scripts/preflight.py')}}
        options.report.parent.mkdir(parents=True,exist_ok=True)
        options.report.write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8', newline='\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
