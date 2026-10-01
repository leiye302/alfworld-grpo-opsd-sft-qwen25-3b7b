"""Protect native output writes from shared file-entry exhaustion, without training changes."""
import errno,json,os,time
from pathlib import Path

class FileEntryReserve:
    def __init__(self, root, target=512):
        self.root=Path(root).resolve()
        self.directory=self.root/'file_entry_reserve'
        self.target=target
        self.event_file=self.root/'logs/storage_guard.jsonl'

    def initialize(self):
        self.directory.mkdir(exist_ok=True)
        assert self.directory.resolve().parent==self.root and not self.directory.is_symlink()
        self.event_file.touch(exist_ok=True)
        return self.maintain(refill=True)

    def available(self):
        return os.statvfs(self.root).f_favail

    def slots(self):
        return sorted(p for p in self.directory.glob('slot_*') if p.name[5:].isdigit())

    def event(self, kind, **values):
        # This file is created before training; appending requires no new entry.
        with self.event_file.open('a') as f:
            f.write(json.dumps({'time':time.time(),'pid':os.getpid(),'kind':kind,**values})+'\n')

    def release(self, count, reason):
        released=0
        for p in self.slots():
            if released>=count:break
            try:
                assert not p.is_symlink() and p.parent.resolve()==self.directory.resolve()
                assert p.stat().st_size==0, str(p)
                p.unlink()
                released+=1
            except FileNotFoundError:
                # Another process may already have released the same slot.
                continue
        if released:self.event('release',count=released,reason=reason)
        return released

    def maintain(self, minimum=128, refill=False):
        free=self.available()
        released=self.release(minimum-free+16,'low_entry_headroom') if free<minimum else 0
        created=0
        present={p.name for p in self.slots()}
        # Replenish only while the shared filesystem has ample free entries.
        if refill and free>self.target+minimum+1024:
            for index in range(self.target):
                name=f'slot_{index:04d}'
                if name in present:continue
                try:
                    fd=os.open(self.directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    os.close(fd);created+=1
                except FileExistsError:pass
                except OSError as e:
                    if e.errno in (errno.EDQUOT,errno.ENOSPC):break
                    raise
        if created:self.event('reserve',count=created)
        return {'free_entries':free,'reserve_entries':len(self.slots()),'released':released,'created':created}

    def retry(self, operation, reason, retries=3):
        for attempt in range(retries+1):
            try:return operation()
            except OSError as e:
                if e.errno!=errno.EDQUOT or attempt==retries:raise
                if not self.release(32,reason):raise
                # Allow filesystem quota accounting to observe our unlink.
                time.sleep(0.2)


def for_run():
    return FileEntryReserve(os.environ['SDAR_RUN_ROOT'])


def install_trainer_hooks(module):
    cls=module.RayPPOTrainer
    if getattr(cls,'_sdar_storage_guard_installed',False):return
    native_dump=cls._dump_generations
    native_save=cls._save_checkpoint

    def dump(self,*args,**kwargs):
        reserve=for_run()
        reserve.maintain()
        # Call the exact native serializer. A failed partial write is reopened
        # with its native "w" mode, so retry neither duplicates nor drops rows.
        return reserve.retry(lambda:native_dump(self,*args,**kwargs),'rollout_output_retry')

    def save(self,*args,**kwargs):
        # Native checkpoint timing, content, collectives and completion marker
        # stay intact. Never retry a distributed checkpoint after partial failure.
        for_run().maintain(minimum=256)
        return native_save(self,*args,**kwargs)

    cls._dump_generations=dump
    cls._save_checkpoint=save
    cls._sdar_storage_guard_installed=True
