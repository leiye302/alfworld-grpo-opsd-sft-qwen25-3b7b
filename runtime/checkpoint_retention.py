"""Verify a native eight-rank save before retiring its previous checkpoint.

No tensor loading, model updates, sampling or distributed calls occur here.
Only the four handoff run directories and the explicitly superseded 50/100
iteration directories may be removed. Native pre-write rotation stays disabled.
"""
from pathlib import Path
import json
import os
import shutil
import time
import zipfile

GROUPS = frozenset(('3b_baseline', '3b_sft', '7b_baseline', '7b_sft'))
SAVE_ITERATIONS = (50, 100, 150)
POLICY = {'save_every': 50, 'iterations': list(SAVE_ITERATIONS),
          'keep': 'latest_complete', 'delete_previous_only_after_verified_save': True}


def _run_root(run):
    run = Path(run).absolute()
    if run.resolve(strict=True) != run or run.name not in GROUPS or run.parent.name != 'runs':
        raise RuntimeError('Checkpoint retention is restricted to the four handoff run directories.')
    directory = run / 'checkpoints'
    if directory.is_symlink() or not directory.is_dir() or directory.resolve() != directory:
        raise RuntimeError('Checkpoint root is missing or redirects outside the run: ' + str(directory))
    return run, directory


def _plain_directory(path, parent):
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True).parent != parent:
        raise RuntimeError('Checkpoint directory is missing or redirects: ' + str(path))
    for item in path.rglob('*'):
        if item.is_symlink():
            raise RuntimeError('Checkpoint contains a symbolic link; nothing will be retired: ' + str(item))


def _torch_archive(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError('Missing or empty native state: ' + str(path))
    # torch.save uses a ZIP container. Its central directory/footer must be
    # readable even for giant shards; do not scan all tensor bytes or unpickle.
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if not any(name.endswith('/data.pkl') for name in names):
                raise RuntimeError('Native state lacks its serialization metadata: ' + str(path))
            if not any(name.endswith('/version') for name in names):
                raise RuntimeError('Native state lacks its serialization version: ' + str(path))
            if any(member.header_offset < 0 or member.header_offset >= archive.start_dir
                   or member.file_size < 0 for member in members):
                raise RuntimeError('Native state has invalid ZIP offsets: ' + str(path))
    except zipfile.BadZipFile as error:
        raise RuntimeError('Native state is truncated or not a torch.save archive: ' + str(path)) from error
    return path.stat().st_size


def verify_checkpoint(run, iteration, world_size=8, require_latest=True):
    """Check rank coverage, closed native archives, tokenizer and latest marker."""
    if iteration not in SAVE_ITERATIONS or world_size != 8:
        raise RuntimeError('This policy only accepts iterations 50/100/150 and eight ranks.')
    run, directory = _run_root(run)
    checkpoint = directory / ('global_step_' + str(iteration))
    _plain_directory(checkpoint, directory)
    actor = checkpoint / 'actor'
    _plain_directory(actor, checkpoint)
    sizes = {'data.pt': _torch_archive(checkpoint / 'data.pt')}
    for kind in ('model', 'optim', 'extra_state'):
        expected = {f'{kind}_world_size_8_rank_{rank}.pt' for rank in range(8)}
        actual = {path.name for path in actor.glob(kind + '_world_size_*_rank_*.pt')}
        if actual != expected:
            raise RuntimeError('Native rank coverage mismatch for ' + kind + ': ' + str(checkpoint))
        for name in sorted(expected):
            sizes['actor/' + name] = _torch_archive(actor / name)
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        path = actor / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Missing native model/tokenizer metadata: ' + str(path))
        json.loads(path.read_text(encoding='utf-8'))
    if require_latest:
        marker = directory / 'latest_checkpointed_iteration.txt'
        if marker.is_symlink() or not marker.is_file() or marker.read_text().strip() != str(iteration):
            raise RuntimeError('Latest-checkpoint marker does not identify the completed save.')
    return {'iteration': iteration, 'world_size': world_size, 'state_files': len(sizes),
            'state_bytes': sum(sizes.values()), 'validation': 'rank/archive/metadata/marker structure; no tensor reload'}


def _event(run, kind, **values):
    path = run / 'logs' / 'checkpoint_retention.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time': time.time(), 'kind': kind, **values}) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def restore_previous_marker(run, iteration):
    """A failed save must not redirect native resume to an incomplete state."""
    run, directory = _run_root(run)
    previous = {100: 50, 150: 100}.get(iteration)
    marker = directory / 'latest_checkpointed_iteration.txt'
    if marker.is_symlink():
        raise RuntimeError('Refusing to replace a symbolic-link latest marker.')
    if previous is None:
        if iteration == 50 and marker.exists() and marker.read_text().strip() == '50':
            marker.unlink()
        return
    verify_checkpoint(run, previous, require_latest=False)
    temporary = directory / 'latest_checkpointed_iteration.txt.retention_tmp'
    # Exclusive creation prevents following a pre-existing redirected file.
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(str(previous))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(marker)
    _event(run, 'failed_save_marker_restored', iteration=iteration, previous=previous)


def after_native_save(run, iteration):
    """Called only once native save returns; errors leave the new save intact."""
    try:
        receipt = verify_checkpoint(run, iteration)
    except Exception:
        restore_previous_marker(run, iteration)
        raise
    run, directory = _run_root(run)
    _event(run, 'save_verified', **receipt)
    previous = {100: 50, 150: 100}.get(iteration)
    if previous is None:
        return receipt
    old = directory / ('global_step_' + str(previous))
    if old.is_symlink():
        raise RuntimeError('Refusing to retire a symbolic-link checkpoint: ' + str(old))
    if not old.exists():
        _event(run, 'previous_already_absent', iteration=iteration, previous=previous)
        return receipt
    _plain_directory(old, directory)
    # Recheck the newest native state and marker immediately before the exact
    # authorized deletion. Never remove sampling evidence or another run.
    verify_checkpoint(run, iteration)
    _event(run, 'retire_previous_start', iteration=iteration, previous=previous)
    shutil.rmtree(old)
    _event(run, 'previous_retired', iteration=iteration, previous=previous)
    return receipt
