"""Shared, timed configuration execution for CLI and interactive entry points."""
import json
import math
import traceback

from accuracy_validator import validate_accuracy
from atc_runner import run_atc
from infer_runner import run_infer
from metric_collector import collect_metrics
from perf_monitor import PerfMonitor
from config_timing import atomic_text, phase


def execute_configuration(connector, journal, exp_id, cfg, onnx, shape,
                          repeat, all_results, results_path='results/results.json'):
    timer = journal.start(exp_id, cfg)
    record = dict(exp_id=exp_id, config=cfg, compile_success=False, metrics={})
    success = False
    interrupted = False
    monitor = None
    with timer.activate():
        try:
            journal.save()
            with phase('compile'):
                journal.save()
                atc = run_atc(exp_id, cfg, onnx, shape)
            record.update(decoded_config=atc.get('decoded_config'),
                          actual_input_shape=atc.get('actual_input_shape'),
                          atc_command=atc.get('command'), compile_success=atc.get('success'))
            if not atc['success']:
                record['compile_error'] = atc.get('error') or 'ATC returned success=False'
                timer.note('编译失败: ' + str(record['compile_error']))
                return record

            journal.save()
            cosine = {'cosine_mean': None, 'cosine_min': None}
            try:
                with phase('validation_measurement'):
                    journal.save()
                    cosine = validate_accuracy(onnx_path=onnx, om_path=atc['om_path'],
                                               input_shape=atc['input_shape_tuple'],
                                               connector=connector, num_samples=10)
            except Exception as exc:
                cosine['error'] = '{0}: {1}'.format(type(exc).__name__, exc)
                timer.note('验证异常: ' + cosine['error'])
                traceback.print_exc()
            record['accuracy_validation'] = cosine

            with phase('preparation'):
                monitor = PerfMonitor(connector, interval=0.5)
                monitor.start()
            journal.save()
            try:
                with phase('validation_measurement'):
                    journal.save()
                    latency, fps, details = run_infer(
                        atc['om_path'], atc['batch_size'], repeat=repeat,
                        raw_log_path='results/logs/msame_exp{0}.log'.format(exp_id))
            except Exception as exc:
                record['inference_error'] = '{0}: {1}'.format(type(exc).__name__, exc)
                timer.note('测量异常: ' + record['inference_error'])
                traceback.print_exc()
                return record

            with phase('result_processing'):
                stopping, monitor = monitor, None
                perf_stats = stopping.stop(raw_log_path='results/logs/npu_smi_exp{0}.log'.format(exp_id))
                metrics = collect_metrics(latency, fps, atc['batch_size'], details)
                metrics.update(cosine_mean=cosine.get('cosine_mean'), cosine_min=cosine.get('cosine_min'),
                               accuracy_validation=cosine, perf_stats=perf_stats)
                record['metrics'] = metrics
                # Execution success is distinct from numerical feasibility (the evaluator applies its threshold).
                values = [latency, fps, cosine.get('cosine_mean'), cosine.get('cosine_min')]
                success = not cosine.get('error') and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in values)
                if not success:
                    timer.note('验证或测量未获得完整有效结果')
                print('Done: Exp{0} latency={1:.6f}ms fps={2:.6f}'.format(exp_id, latency, fps), flush=True)
            return record
        except BaseException as exc:
            success = False
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            message = '{0}: {1}'.format(type(exc).__name__, exc)
            record['execution_error'] = message
            timer.note('{0}阶段{1}: {2}'.format(timer.error_stage or timer.stage,
                                             '中断' if interrupted else '异常', message))
            raise
        finally:
            timer.switch('result_processing')
            try:
                if monitor is not None:
                    try:
                        monitor.stop(raw_log_path='results/logs/npu_smi_exp{0}.log'.format(exp_id))
                    except Exception as exc:
                        success = False
                        timer.note('监控清理异常: {0}: {1}'.format(type(exc).__name__, exc))
                all_results.append(record)
                # Include result serialization and disk write in result-processing time.
                atomic_text(results_path, json.dumps(all_results, ensure_ascii=False, indent=2))
            except BaseException as exc:
                success = False
                interrupted = interrupted or isinstance(exc, (KeyboardInterrupt, SystemExit))
                timer.note('结果保存/清理异常: {0}: {1}'.format(type(exc).__name__, exc))
                raise
            finally:
                timer.finish(success, interrupted)
                journal.save()
                print('[timing] {0}: {1:.3f}s ({2}); {3}'.format(
                    timer.config_id, timer.snapshot()['total_seconds'], timer.status,
                    journal.directory / 'config_timing.csv'), flush=True)
