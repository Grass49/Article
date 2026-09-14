"""Exclusive wall-clock timing; no hardware dependencies and no metric changes."""
import csv
import io
import json
import os
import signal
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path

_active = ContextVar('config_timing', default=None)
STAGES = ('compile', 'preparation', 'validation_measurement', 'result_processing')
COLUMNS = ('配置ID', '编译秒数', '准备与初始化秒数', '验证与测量秒数',
           '结果处理秒数', '配置总耗时秒数', '是否成功', '中断或异常备注',
           '运行ID', '实验序号', '状态', '当前阶段', '开始时间', '结束时间')


def atomic_text(path, text, encoding='utf-8'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding=encoding, newline='') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def phase(name):
    timer = _active.get()
    if timer is None:
        yield
        return
    previous = timer.stage
    timer.switch(name)
    try:
        yield
    except BaseException:
        if timer.error_stage is None:
            timer.error_stage = name
        raise
    finally:
        timer.switch(previous)


def timed(name):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with phase(name):
                return func(*args, **kwargs)
        return wrapped
    return decorate


def add_timing_note(message):
    timer = _active.get()
    if timer is not None:
        timer.note(message)


def install_interrupt_handlers():
    """Allow SIGTERM to unwind finally blocks, like Ctrl+C (SIGINT)."""
    def terminate(signum, frame):
        raise KeyboardInterrupt('收到终止信号 {0}'.format(signum))
    signal.signal(signal.SIGTERM, terminate)


class TimingJournal:
    def __init__(self, output_dir='results'):
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        # Each invocation keeps its own audit trail; prior runs are never overwritten.
        self.directory = Path(output_dir) / 'timing' / self.run_id
        self.rows = []

    def start(self, exp_id, cfg):
        timer = ConfigTimer(self, exp_id, cfg)
        self.rows.append(timer)
        return timer

    def save(self):
        records = [timer.snapshot() for timer in self.rows]
        atomic_text(self.directory / 'config_timing.json', json.dumps(records, ensure_ascii=False, indent=2))
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for r in records:
            writer.writerow([r['config_id']] + [r[k + '_seconds'] for k in STAGES]
                            + [r['total_seconds'], '' if r['success'] is None else ('是' if r['success'] else '否'),
                               r['notes'], r['run_id'], r['exp_id'], r['status'], r['stage'],
                               r['started_at'], r['finished_at']])
        atomic_text(self.directory / 'config_timing.csv', stream.getvalue(), 'utf-8-sig')


class ConfigTimer:
    def __init__(self, journal, exp_id, cfg):
        self.journal = journal
        self.exp_id = exp_id
        self.cfg = dict(cfg)
        self.config_id = '-'.join(k + str(cfg[k]) for k in 'ABCD')
        self.stage = 'preparation'
        self.error_stage = None
        self.started = self.last = time.perf_counter()
        self.started_at = datetime.now().astimezone().isoformat()
        self.finished_at = ''
        self.elapsed = dict.fromkeys(STAGES, 0.0)
        self.messages = []
        self.success = None
        self.status = 'running'
        self.end = None

    def switch(self, name):
        if name not in STAGES:
            raise ValueError('Unknown timing phase: ' + str(name))
        now = time.perf_counter()
        self.elapsed[self.stage] += now - self.last
        self.last = now
        self.stage = name

    def note(self, text):
        if text and str(text) not in self.messages:
            self.messages.append(str(text))

    def snapshot(self):
        now = self.end if self.end is not None else time.perf_counter()
        elapsed = dict(self.elapsed)
        if self.end is None:
            elapsed[self.stage] += now - self.last
        r = dict(run_id=self.journal.run_id, exp_id=self.exp_id, config_id=self.config_id,
                 config=self.cfg, started_at=self.started_at, finished_at=self.finished_at,
                 status=self.status, stage=self.stage, success=self.success,
                 notes=' | '.join(self.messages), total_seconds=round(now - self.started, 6))
        r.update({k + '_seconds': round(v, 6) for k, v in elapsed.items()})
        return r

    def finish(self, success, interrupted=False):
        now = time.perf_counter()
        self.elapsed[self.stage] += now - self.last
        self.last = self.end = now
        self.finished_at = datetime.now().astimezone().isoformat()
        self.success = bool(success)
        self.status = 'interrupted' if interrupted else ('success' if success else 'failed')

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)
