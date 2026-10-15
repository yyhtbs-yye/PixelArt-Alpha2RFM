import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from scripts import train_rfm


@pytest.fixture
def launch_workspace(tmp_path, monkeypatch):
    source = Path(train_rfm.__file__).resolve().parents[1]
    for name in ('scripts', 'datasets', 'diffusion', 'configs'):
        shutil.copytree(source / name, tmp_path / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source / 'requirements.txt', tmp_path / 'requirements.txt')
    smoke = tmp_path / 'output/rfm_mix50_smoke'
    smoke.mkdir(parents=True)
    (smoke / 'verification.json').write_text(json.dumps(dict(passed=True)))
    monkeypatch.setattr(train_rfm, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.parametrize('mode', ['--launch', '--controller'])
def test_launch_modes_dispatch_with_config_overrides(launch_workspace, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(train_rfm, 'launch_training', lambda args, config: calls.append((args, config)))
    monkeypatch.setattr(sys, 'argv', ['train_rfm.py', 'configs/rfm.py', mode,
                                    '--work-dir', 'output/custom', '--max-train-steps', '123'])
    train_rfm.main()
    assert len(calls) == 1
    args, config = calls[0]
    assert config['work_dir'] == 'output/custom' and config['max_train_steps'] == 123
    assert args.controller == (mode == '--controller')


@pytest.mark.parametrize('arguments', [[], ['--launch', '--resume', 'checkpoint.pth'], ['--launch', '--controller']])
def test_invalid_launch_combinations_fail_before_side_effects(launch_workspace, monkeypatch, arguments):
    monkeypatch.setattr(sys, 'argv', ['train_rfm.py', *arguments])
    with pytest.raises(SystemExit) as error:
        train_rfm.main()
    assert error.value.code == 2
    assert not (launch_workspace / 'output/rfm_mix50').exists()


def test_launch_rejects_distributed_worker_environment(launch_workspace, monkeypatch):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setattr(sys, 'argv', ['train_rfm.py', '--launch'])
    with pytest.raises(SystemExit) as error:
        train_rfm.main()
    assert error.value.code == 2
    assert not (launch_workspace / 'output/rfm_mix50').exists()


def test_torchrun_mode_enters_training_without_launching(launch_workspace, monkeypatch):
    for name, value in dict(RANK='0', LOCAL_RANK='0', WORLD_SIZE='4').items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, 'argv', ['train_rfm.py', 'configs/rfm.py'])
    reached = []

    def stop_before_cuda(device):
        reached.append(device)
        raise RuntimeError('training dispatch reached')

    monkeypatch.setattr(train_rfm.torch.cuda, 'set_device', stop_before_cuda)
    with pytest.raises(RuntimeError, match='training dispatch reached'):
        train_rfm.main()
    assert reached == [0]


def test_background_launch_uses_same_entrypoint_and_forwards_overrides(launch_workspace, monkeypatch):
    calls = []

    def start_controller(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(train_rfm.subprocess, 'Popen', start_controller)
    monkeypatch.setattr(sys, 'argv', ['train_rfm.py', '--launch', '--work-dir', 'output/custom',
                                    '--max-train-steps', '123', '--validate-every-updates', '7',
                                    '--save-every-updates', '9'])
    train_rfm.main()
    assert len(calls) == 1
    command, options = calls[0]
    assert command[:4] == [sys.executable, '-u', 'scripts/train_rfm.py', 'configs/rfm.py']
    assert '--controller' in command and '--launch' not in command
    for flag, value in [('--work-dir', 'output/custom'), ('--max-train-steps', '123'),
                        ('--validate-every-updates', '7'), ('--save-every-updates', '9')]:
        assert command[command.index(flag) + 1] == value
    assert options['cwd'] == launch_workspace and options['start_new_session']
    work = launch_workspace / 'output/custom'
    saved = json.loads((work / 'config.json').read_text())
    assert saved['max_train_steps'] == 123 and saved['validate_every_updates'] == 7
    hashes = json.loads((work / 'preparation/source_sha256.json').read_text())
    assert 'scripts/train_rfm.py' in hashes
    for name, expected in hashes.items():
        assert hashlib.sha256((work / 'preparation/source_snapshot' / name).read_bytes()).hexdigest() == expected


def test_controller_prepares_caches_then_starts_torchrun_without_recursion(launch_workspace, monkeypatch):
    config = train_rfm.runpy.run_path('configs/rfm.py')['config']
    work = launch_workspace / 'output/custom'
    preparation = work / 'preparation'
    preparation.mkdir(parents=True)
    (preparation / 'source_sha256.json').write_text(json.dumps({
        'scripts/train_rfm.py': hashlib.sha256((launch_workspace / 'scripts/train_rfm.py').read_bytes()).hexdigest(),
    }))
    for source in config['source_datasets']:
        source = Path(source)
        source.mkdir(parents=True)
        (source / 'metadata.jsonl').write_text('{}\n')
        pilot_name = 'localized_narratives' if 'localized_narratives' in source.name else 'flux1_dev'
        pilot = launch_workspace / 'output/rfm_mix50_smoke/cache' / pilot_name
        pilot.mkdir(parents=True)
        (pilot / 'cache_settings.json').write_text(json.dumps(dict(
            source=str(source.resolve()),
            source_metadata_sha256=hashlib.sha256((source / 'metadata.jsonl').read_bytes()).hexdigest())))
    calls = []

    def start_child(command, **kwargs):
        calls.append(command)
        if 'torch.distributed.run' in command:
            (work / 'completed.json').write_text(json.dumps(dict(optimizer_step=123)))
        return SimpleNamespace(pid=12345, poll=lambda: 0)

    def finalize_cache(command, **kwargs):
        assert '--finalize' in command and kwargs['check']
        cache = Path(command[command.index('--output') + 1])
        index = config['data_roots'].index(str(cache))
        (cache / 'summary.json').write_text(json.dumps(dict(complete=True, images=config['source_counts'][index])))

    monkeypatch.setattr(train_rfm.shutil, 'disk_usage', lambda path: SimpleNamespace(free=400 * 2**30))
    monkeypatch.setattr(train_rfm.subprocess, 'Popen', start_child)
    monkeypatch.setattr(train_rfm.subprocess, 'run', finalize_cache)
    monkeypatch.setattr(sys, 'argv', ['train_rfm.py', '--controller', '--work-dir', 'output/custom',
                                    '--max-train-steps', '123'])
    train_rfm.main()
    trainers = [command for command in calls if 'torch.distributed.run' in command]
    assert len(trainers) == 1 and len(calls) == 9
    command = trainers[0]
    assert command[command.index('scripts/train_rfm.py') + 1] == 'configs/rfm.py'
    assert '--launch' not in command and '--controller' not in command
    assert command[command.index('--work-dir') + 1] == 'output/custom'
    assert command[command.index('--max-train-steps') + 1] == '123'
    assert json.loads((preparation / 'status.json').read_text())['stage'] == 'completed'
