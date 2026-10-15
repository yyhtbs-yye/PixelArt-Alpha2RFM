import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def launcher_environment(tmp_path):
    record = tmp_path / 'command.json'
    python = tmp_path / 'python with spaces'
    python.write_text(f'#!{sys.executable}\n' + '''import json
import os
from pathlib import Path
import sys
if sys.argv[1] == '-':
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), '<launcher config reader>', 'exec'))
else:
    Path(os.environ['RFM_COMMAND_RECORD']).write_text(json.dumps(dict(
        args=sys.argv[1:], cwd=os.getcwd(), cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))))
''')
    python.chmod(0o755)
    config = tmp_path / 'config with spaces.py'
    config.write_text("config = dict(world_size=3)\n")
    environment = dict(os.environ, PYTHON_BIN=str(python), RFM_COMMAND_RECORD=str(record),
                       CUDA_VISIBLE_DEVICES='2,3,5')
    return config, record, environment


@pytest.mark.parametrize('mode', ['train', 'validate', 'inference'])
def test_launcher_dispatches_with_config_and_preserves_arguments(tmp_path, launcher_environment, mode):
    config, record, environment = launcher_environment
    root = Path(__file__).resolve().parents[1]
    extras = ['--work-dir', 'output/with spaces'] if mode == 'train' else [
        '--checkpoint', 'weights/model with spaces.pth', '--output', 'output/with spaces']
    if mode == 'inference':
        extras += ['--prompts', 'prompts with spaces.json', '--weights', 'ema']
    result = subprocess.run(['bash', str(root / 'run.sh'), mode, str(config), *extras],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = json.loads(record.read_text())
    if mode == 'train':
        expected = ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=3',
                    'scripts/train_rfm.py', str(config), *extras]
    else:
        expected = ['-m', f'scripts.{mode}_rfm', '--config', str(config), *extras]
    assert command['args'] == expected
    assert command['cwd'] == str(root) and command['cuda'] == '2,3,5'


@pytest.mark.parametrize('flag', ['--launch', '--help'])
def test_launch_and_help_do_not_start_torchrun(tmp_path, launcher_environment, flag):
    config, record, environment = launcher_environment
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', str(root / 'run.sh'), 'train', str(config), flag],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text())['args'] == ['-m', 'scripts.train_rfm', str(config), flag]


@pytest.mark.parametrize('invalid', ['action', 'config', 'world_size'])
def test_invalid_launcher_inputs_do_not_start_a_process(tmp_path, launcher_environment, invalid):
    config, record, environment = launcher_environment
    root = Path(__file__).resolve().parents[1]
    action = 'train'
    if invalid == 'action':
        action = 'unknown'
    elif invalid == 'config':
        config = tmp_path / 'missing.py'
    else:
        config.write_text('config = dict(world_size=0)\n')
    result = subprocess.run(['bash', str(root / 'run.sh'), action, str(config)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert not record.exists()
