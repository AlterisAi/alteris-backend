# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Alteris CLI binary.

Usage (from repo root):
    source .venv/bin/activate
    pyinstaller scripts/dist/alteris-cli.spec --noconfirm \
        --distpath build/dist --workpath build/pyinstaller
"""
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    'alteris.adapters.mail',
    'alteris.adapters.imessage',
    'alteris.adapters.whatsapp',
    'alteris.adapters.calendar',
    'EventKit',
    'alteris.adapters.contacts',
    'alteris.adapters.granola',
    'alteris.adapters.slack',
    'alteris.llm.ollama',
    'alteris.llm.gemini',
    'alteris.llm.mock',
    'alteris.eval',
    'alteris.eval.checks',
    'alteris.eval.sampler',
    'alteris.eval.reviewer',
    'alteris.eval.runner',
    'alteris.eval.stats',
    'alteris.eval.golden',
    'tzdata',
    'pydantic',
    'requests',
    'google.genai',
    'slack_sdk',
    'watchdog',
]

tmp_ret = collect_all('alteris')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tzdata')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyobjc-framework-EventKit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Use SPECPATH so the entry point resolves relative to the repo, not /tmp
# SPECPATH = .../alteris-backend/scripts/dist → 2 dirname calls to repo root
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(SPECPATH)))
CLI_ENTRY = os.path.join(REPO, 'src', 'alteris', 'cli.py')

a = Analysis(
    [CLI_ENTRY],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['numpy', 'scipy', 'matplotlib', 'PIL', 'tkinter', 'test'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='alteris-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='alteris-cli',
)
