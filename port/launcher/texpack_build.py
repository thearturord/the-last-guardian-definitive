"""Texture pack build from the launcher (experimental): fetches a ready-made generator kit
(ps4-texturepack-generator: Python + PyTorch + texconv, nothing installed), the upscale model from its
author, and runs `texpack.py make` on the player's own game copy.

Only the FULL build for now: every texture upscaled (`--all --max-size 8192`). A light build (only the
textures the game draws larger than they are) is planned and will be the advised one.

No Tk in here: the launcher's dialog calls these from a worker thread and shows `progress` callbacks.
"""

import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# Kits are release assets of the generator's repository; TLG_TEXPACK_KIT_URL points elsewhere for
# tests (a URL or a local folder with the same files).
KIT_RELEASE = 'kit-v1'
KIT_URL = f'https://github.com/tabnul/ps4-texturepack-generator/releases/download/{KIT_RELEASE}'
SYSTEM = 'windows' if sys.platform == 'win32' else 'linux'
GPUS = ('nvidia', 'amd', 'cpu')
# FULL build: about 77 GB of PNGs while upscaling plus about 77 GB of DDS pack; --clean removes the PNGs
# at the end. Measured on the TLG v2 pack (3934 textures).
FULL_PEAK_GB = 160
FULL_PACK_GB = 80
FULL_ARGS = ['--all', '--max-size', '8192', '--clean']
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def detect_gpu(run=subprocess.run):
    """'nvidia' (RTX 20 or newer), 'nvidia-old' (GTX 10xx and older: the CUDA 12.8 kit does not run on
    them), 'amd', or None when unknown. Slow (a process): off the UI thread."""
    try:
        out = run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True,
                  text=True, timeout=15, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW).stdout
        caps = [float(line) for line in out.split() if line.replace('.', '', 1).isdigit()]
        if caps:
            return 'nvidia' if max(caps) >= 7.5 else 'nvidia-old'
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if SYSTEM == 'linux':
        for vendor in Path('/sys/class/drm').glob('card*/device/vendor'):
            try:
                if vendor.read_text().strip() == '0x1002':
                    return 'amd'
            except OSError:
                pass
    return None


def default_kit_gpu(detected):
    """The kit to offer first. Unknown on Windows: DirectML, which also runs on Intel."""
    if detected == 'nvidia' or detected == 'amd':
        return detected
    return 'amd' if SYSTEM == 'windows' and detected is None else 'cpu'


def kit_name(gpu):
    return f'kit-{SYSTEM}-{gpu}'


def kit_dir(root, gpu):
    return Path(root) / kit_name(gpu)


def installed_kit(root, gpu):
    """kit.json of an unpacked kit, or None."""
    try:
        return json.loads((kit_dir(root, gpu) / 'kit.json').read_text())
    except (OSError, ValueError):
        return None


def _open(url_or_path):
    base = os.environ.get('TLG_TEXPACK_KIT_URL', KIT_URL).rstrip('/')
    where = f'{base}/{url_or_path}'
    if '://' not in base:
        return open(where, 'rb')
    request = urllib.request.Request(where, headers={'User-Agent': 'tlg-launcher'})
    return urllib.request.urlopen(request, timeout=60)


def fetch_manifest(gpu):
    with _open(f'{kit_name(gpu)}.json') as src:
        return json.loads(src.read())


def free_gb(path):
    """Free space where `path` is (or would be), in GB."""
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def needed_gb(manifest, kit_ready):
    """(GB on the kit's drive, GB on the pack's drive) a FULL build needs."""
    kit = 0 if kit_ready else (manifest['archive_size'] + manifest['unpacked_size']) / 1e9
    return kit, FULL_PEAK_GB


def _download(src, dest, size, sha, progress, label, stop):
    h, done = hashlib.sha256(), 0
    with open(dest, 'wb') as out:
        while chunk := src.read(1 << 20):
            if stop():
                raise InterruptedError
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            progress(label, done / size if size else 0)
    if h.hexdigest() != sha:
        dest.unlink()
        raise ValueError(f'{dest.name}: download damaged (sha256 does not match)')


class _Chain:
    """The kit parts read one after the other, as one file."""

    def __init__(self, paths):
        self.paths, self.file = list(paths), None

    def read(self, n=-1):
        while self.paths or self.file:
            if self.file is None:
                self.file = open(self.paths.pop(0), 'rb')
            data = self.file.read(n)
            if data:
                return data
            self.file.close()
            self.file = None
        return b''

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def install_kit(root, gpu, manifest, progress, stop=lambda: False):
    """Downloads (parts already there with the right sha256 are kept), checks and unpacks a kit."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts = []
    for number, part in enumerate(manifest['parts']):
        dest = root / part['name']
        label = f'kit {number + 1}/{len(manifest["parts"])}'
        if not (dest.is_file() and dest.stat().st_size == part['size'] and _sha256(dest) == part['sha256']):
            with _open(part['name']) as src:
                _download(src, dest, part['size'], part['sha256'], progress, label, stop)
        parts.append(dest)
    target = kit_dir(root, gpu)
    shutil.rmtree(target, ignore_errors=True)
    progress('unpack', None)
    chain = _Chain(parts)
    try:
        with tarfile.open(fileobj=chain, mode='r|xz') as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(target, filter='data')
            else:
                tar.extractall(target)
    finally:
        chain.close()
    for part in parts:
        part.unlink()
    return installed_kit(root, gpu)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def model_path(root, gpu, kit):
    return kit_dir(root, gpu) / kit['model_dir'] / kit['model']['name']


def install_model(root, gpu, kit, progress, stop=lambda: False):
    """The upscale model from its author's page (not in the kit: its license is its author's)."""
    model = kit['model']
    dest = model_path(root, gpu, kit)
    if dest.is_file() and dest.stat().st_size == model['size'] and _sha256(dest) == model['sha256']:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(model['url'], headers={'User-Agent': 'tlg-launcher'})
    with urllib.request.urlopen(request, timeout=60) as src:
        _download(src, dest, model['size'], model['sha256'], progress, 'model', stop)
    return dest


def build_command(root, gpu, kit, game_dir, pack_dir):
    kit_root = kit_dir(root, gpu)
    # The GPU kits share a card: one process on AMD (DirectML/ROCm untested with more) and on the CPU.
    jobs = '3' if gpu == 'nvidia' else '1'
    return [str(kit_root / kit['python_exe']), '-E', '-s', str(kit_root / kit['script']), 'make',
            str(game_dir), str(pack_dir), *FULL_ARGS, '--jobs', jobs]


def start_build(command, log_path):
    """Starts the build with its output in log_path; its own process group so Stop ends all workers."""
    log = open(log_path, 'a', encoding='utf-8', errors='replace')
    kwargs = {'creationflags': NO_WINDOW | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)} \
        if SYSTEM == 'windows' else {'start_new_session': True}
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONUNBUFFERED'] = '1'
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               cwd=Path(command[3]).parent, env=env, **kwargs)
    log.close()
    return process


def stop_build(process):
    """Ends the build and its worker processes. Started again later, it continues where it stopped."""
    if process.poll() is not None:
        return
    if SYSTEM == 'windows':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(process.pid)], capture_output=True,
                       creationflags=NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass


def build_progress(pack_dir):
    """(phase, done, total) from the files on disk: 'scan' before the list exists, then 'upscale'
    (PNGs made) and 'encode' (DDS made). Works for a resumed build too."""
    pack_dir = Path(pack_dir)
    work = pack_dir.with_name(pack_dir.name + '.work')
    try:
        with open(work / 'manifest.csv', encoding='utf-8', newline='') as f:
            total = sum(1 for row in csv.DictReader(f) if row.get('decision') == 'pack')
    except (OSError, csv.Error):
        return 'scan', 0, 0
    dds = sum(1 for _ in pack_dir.rglob('*.dds')) if pack_dir.is_dir() else 0
    if dds:
        return 'encode', dds, total
    pngs = sum(1 for _ in (work / 'png').rglob('*.png')) if (work / 'png').is_dir() else 0
    return 'upscale', pngs, total
