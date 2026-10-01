"""Retry quota-failed file creation using only this run's expendable empty slots.

Does not alter tensors, sampling, checkpoints, or successful filesystem calls.
Installed per experiment, never in the shared Python installation.
"""
import builtins
import errno
import functools
import io
import os
from pathlib import Path
import threading
import time

BASE = '/mnt/zixuan/test/VLA_test'
_local = threading.local()
_installed = False
# Shared quota accounting can lag unlinks. Only failed filesystem operations
# wait; the ordinary successful path and training math are unchanged.
RETRY_DELAYS = (.1, .2, .3, .4, 1., 2., 4., 8., 15., 30.)


def within_scope(path):
    try:
        path = os.fsdecode(os.fspath(path))
        return os.path.commonpath((os.path.realpath(path), BASE)) == BASE
    except (TypeError, ValueError, OSError):
        return False


def release_slots(count=32):
    root = Path(os.environ['SDAR_RUN_ROOT']).resolve()
    assert root.parent == Path(BASE) and not root.is_symlink()
    directory = root/'file_entry_reserve'
    assert directory.resolve() == directory and not directory.is_symlink()
    released = 0
    for path in sorted(directory.glob('slot_*')):
        if released >= count:
            break
        if not path.name[5:].isdigit():
            continue
        try:
            if path.is_symlink() or path.stat().st_size != 0:
                continue
            path.unlink()
            released += 1
        except FileNotFoundError:
            continue
    try:
        os.write(2, f'[QUOTA_RETRY] pid={os.getpid()} released_empty_slots={released}\n'.encode())
    except OSError:
        pass
    return released


def retry(call, path):
    # No pre-call stat and no changed arguments on the ordinary successful path.
    try:
        return call()
    except OSError as first:
        if first.errno not in (errno.EDQUOT, errno.ENOSPC) or getattr(_local,'busy',False) or not within_scope(path):
            raise
        # Never consume emergency slots in order to create more empty slots.
        # statvfs is advisory and may lag the actual per-operation quota check.
        if 'file_entry_reserve' in Path(os.path.realpath(os.fsdecode(os.fspath(path)))).parts:
            raise
        _local.busy = True
        try:
            for attempt, delay in enumerate(RETRY_DELAYS):
                # A peer may have freed the same slots: still retry the operation.
                release_slots()
                time.sleep(delay)
                try:
                    return call()
                except OSError as error:
                    if error.errno not in (errno.EDQUOT, errno.ENOSPC):
                        raise
                    if attempt == len(RETRY_DELAYS)-1:
                        raise
        finally:
            _local.busy = False


def wrap_file_open(original):
    @functools.wraps(original)
    def opened(file, *args, **kwargs):
        mode = kwargs.get('mode',args[0] if args else 'r')
        if isinstance(mode,str) and any(c in mode for c in 'wax+'):
            return retry(lambda:original(file,*args,**kwargs), file)
        return original(file,*args,**kwargs)
    return opened


def wrap_os_open(original):
    @functools.wraps(original)
    def opened(path, flags, *args, **kwargs):
        if flags & (os.O_CREAT|os.O_WRONLY|os.O_RDWR):
            return retry(lambda:original(path,flags,*args,**kwargs), path)
        return original(path,flags,*args,**kwargs)
    return opened


def install():
    global _installed
    if _installed:
        return
    _installed=True
    builtins.open=wrap_file_open(builtins.open)
    io.open=wrap_file_open(io.open)
    os.open=wrap_os_open(os.open)
    mkdir=os.mkdir
    def guarded_mkdir(path,*args,**kwargs):
        return retry(lambda:mkdir(path,*args,**kwargs),path)
    os.mkdir=guarded_mkdir
    for name in ('rename','replace'):
        original=getattr(os,name)
        def guarded_move(source,destination,*args,_original=original,**kwargs):
            return retry(lambda:_original(source,destination,*args,**kwargs),destination)
        setattr(os,name,guarded_move)
