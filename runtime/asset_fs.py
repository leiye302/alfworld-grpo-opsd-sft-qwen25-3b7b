"""Read-only ALFWorld archive filesystem, restricted to one declared asset root.

The simulator, task selection, rewards, game JSON and protocol are unmodified.
Only filesystem access to exact archived bytes is virtualized to avoid inode quota.
"""
import builtins
import io
import os
import sqlite3
import threading
from pathlib import PurePosixPath

def install():
    root = os.environ.get('ALFWORLD_DATA', '').rstrip('/')
    database = os.environ.get('SDAR_ALFWORLD_SQLITE', '')
    if not root or not database:
        return
    original_open, original_io_open = builtins.open, io.open
    original_walk = os.walk
    original_exists, original_isfile, original_isdir = os.path.exists, os.path.isfile, os.path.isdir
    db = sqlite3.connect('file:'+database+'?mode=ro&immutable=1', uri=True)
    # Row order records archive member order. Do not sort/reseed game selection.
    members = [r[0] for r in db.execute('SELECT path FROM files ORDER BY rowid')]
    db.close()
    files = set(members)
    tree = {'': [[], []]}
    for name in members:
        parts = PurePosixPath(name).parts
        parent = ''
        for part in parts[:-1]:
            child = parent+'/'+part if parent else part
            if child not in tree:
                tree[child] = [[], []]
                tree[parent][0].append(part)
            parent = child
        tree[parent][1].append(parts[-1])
    local = threading.local()

    def key(path):
        if isinstance(path, int):
            return None
        try:
            path = os.path.abspath(os.fsdecode(path))
        except TypeError:
            return None
        if path == root:
            return ''
        return path[len(root)+1:] if path.startswith(root+'/') else None

    def read(name):
        if getattr(local, 'pid', None) != os.getpid():
            local.connection = sqlite3.connect('file:'+database+'?mode=ro&immutable=1', uri=True)
            local.pid = os.getpid()
        row = local.connection.execute('SELECT data FROM files WHERE path=?', (name,)).fetchone()
        if row is None:
            raise FileNotFoundError(root+'/'+name)
        return row[0]

    def opened(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None,
               closefd=True, opener=None, _original=original_open):
        name = key(path)
        if name is None:
            return _original(path, mode, buffering, encoding, errors, newline, closefd, opener)
        if any(x in mode for x in 'wax+'):
            raise PermissionError('ALFWorld archive is read-only: '+str(path))
        stream = io.BytesIO(read(name))
        stream.name = os.fspath(path)
        return stream if 'b' in mode else io.TextIOWrapper(stream, encoding=encoding or 'utf-8', errors=errors, newline=newline)

    def exists(path):
        k = key(path)
        return (k in files or k in tree) if k is not None else original_exists(path)

    def isfile(path):
        k = key(path)
        return k in files if k is not None else original_isfile(path)

    def isdir(path):
        k = key(path)
        return k in tree if k is not None else original_isdir(path)

    def walk(top, topdown=True, onerror=None, followlinks=False):
        k = key(top)
        if k is None:
            yield from original_walk(top, topdown, onerror, followlinks)
            return
        if k not in tree:
            return
        directories, entries = (list(x) for x in tree[k])
        if topdown:
            yield os.fspath(top), directories, entries
        for directory in directories:
            yield from walk(os.path.join(top, directory), topdown, onerror, followlinks)
        if not topdown:
            yield os.fspath(top), directories, entries

    builtins.open = opened
    io.open = lambda *a, **kw: opened(*a, _original=original_io_open, **kw)
    os.path.exists, os.path.isfile, os.path.isdir, os.walk = exists, isfile, isdir, walk

