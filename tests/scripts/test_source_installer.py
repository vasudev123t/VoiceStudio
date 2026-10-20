"""Root source-setup installers (install.sh / install.ps1): slow-network behaviour.

Pinned from a real Windows run on a ~4 MB/s link: uv's 30 s default timeout
killed Applio's CUDA wheel download, Applio's run-install.bat still reported
success, and the env had no torch. The installers now raise uv's timeout and
retry count (keeping values the user set), and retry Applio's dependency
install with the env's own interpreter before giving up.

install.sh is sourced with VOICESTUDIO_INSTALL_SOURCE_ONLY=1 and its helpers
run against a fake Applio checkout plus a `uv` shim. install.ps1 helpers are
extracted from the script's AST and run with mocks when pwsh is available.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SH = os.path.join(REPO, "install.sh")
PS1 = os.path.join(REPO, "install.ps1")

BASH = shutil.which("bash")
PWSH = shutil.which("pwsh")

_ENV_KEYS = ("UV_HTTP_TIMEOUT", "UV_HTTP_RETRIES", "OMNIVOICE_CUDNN8_TIMEOUT",
             "HF_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR")


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    env["NO_COLOR"] = "1"
    env.update(extra)
    return env


def _write_exe(path, body):
    with open(path, "w") as fh:
        fh.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _source(snippet, args=(), env=None):
    script = (
        'VOICESTUDIO_INSTALL_SOURCE_ONLY=1 source "$0" "$@"\n'
        + textwrap.dedent(snippet)
    )
    return subprocess.run(
        [BASH, "-c", script, SH, *args],
        capture_output=True, text=True, env=env or _clean_env(), timeout=60,
    )


# ── install.sh ──────────────────────────────────────────────────────────────
needs_bash = pytest.mark.skipif(BASH is None or os.name == "nt", reason="needs a POSIX bash")


@needs_bash
def test_sh_sourcing_runs_no_steps():
    r = _source("echo sourced-ok")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("sourced-ok")
    assert "==>" not in r.stdout


@needs_bash
def test_sh_network_defaults_raise_uv_timeout_and_retries():
    r = _source('apply_network_defaults >/dev/null; echo "$UV_HTTP_TIMEOUT $UV_HTTP_RETRIES $OMNIVOICE_CUDNN8_TIMEOUT"')
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-3:] == ["600", "5", "1800"]


@needs_bash
def test_sh_network_defaults_keep_user_values():
    env = _clean_env(UV_HTTP_TIMEOUT="42", UV_HTTP_RETRIES="9", OMNIVOICE_CUDNN8_TIMEOUT="77")
    r = _source('apply_network_defaults >/dev/null; echo "$UV_HTTP_TIMEOUT $UV_HTTP_RETRIES $OMNIVOICE_CUDNN8_TIMEOUT"', env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-3:] == ["42", "9", "77"]


@needs_bash
def test_sh_cache_dir_sets_hf_and_uv_caches(tmp_path):
    cache = tmp_path / "cache"
    r = _source('apply_network_defaults >/dev/null; printf "%s\\n" "$HF_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"',
                args=("--cache-dir", str(cache)))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-3:] == [str(cache / "huggingface"), str(cache / "uv"), str(cache / "pip")]
    assert cache.is_dir()


def _fake_applio(tmp_path, fail_times):
    """Applio checkout whose env imports only after `uv pip install -r` succeeded."""
    applio = tmp_path / "applio"
    (applio / "rvc").mkdir(parents=True)
    (applio / "core.py").write_text("")
    (applio / "requirements.txt").write_text("torch==2.11.0\n")
    (applio / ".venv" / "bin").mkdir(parents=True)
    marker = tmp_path / "deps-installed"
    _write_exe(applio / ".venv" / "bin" / "python", f'[ -f "{marker}" ]\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "uv-calls"
    _write_exe(bin_dir / "uv", textwrap.dedent(f"""\
        echo "$*" >> "{calls}"
        n=$(wc -l < "{calls}")
        if [ "$n" -le {fail_times} ]; then echo "error: network timeout" >&2; exit 2; fi
        touch "{marker}"
        """))
    return applio, bin_dir, calls


def _run_deps(tmp_path, fail_times, variant="cuda", os_name="Linux"):
    applio, bin_dir, calls = _fake_applio(tmp_path, fail_times)
    env = _clean_env(PATH=f"{bin_dir}:{os.environ['PATH']}", APPLIO_RETRY_DELAY="0")
    r = _source(f"""
        APPLIO_DIR={applio}; RESOLVED_VARIANT={variant}; OS={os_name}
        if applio_install_deps; then echo RESULT=ok; else echo RESULT=fail; fi
    """, env=env)
    lines = calls.read_text().splitlines() if calls.exists() else []
    return r, lines, applio


@needs_bash
def test_sh_applio_deps_retry_until_importable(tmp_path):
    r, calls, applio = _run_deps(tmp_path, fail_times=2)
    assert "RESULT=ok" in r.stdout, r.stdout + r.stderr
    assert len(calls) == 3
    for line in calls:
        assert line.startswith(f"pip install --python {applio}/.venv/bin/python -r requirements.txt")
        assert "--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match" in line
    assert "attempt 3/3" in r.stdout


@needs_bash
def test_sh_applio_deps_give_up_after_three_attempts(tmp_path):
    r, calls, _ = _run_deps(tmp_path, fail_times=99)
    assert "RESULT=fail" in r.stdout, r.stdout + r.stderr
    assert len(calls) == 3


@needs_bash
@pytest.mark.parametrize("variant,os_name,expected", [
    ("cpu", "Linux", "--extra-index-url https://download.pytorch.org/whl/cpu"),
    ("rocm", "Linux", "--extra-index-url https://download.pytorch.org/whl/rocm6.4"),
    ("native", "Darwin", None),
])
def test_sh_applio_deps_use_the_variant_index(tmp_path, variant, os_name, expected):
    r, calls, _ = _run_deps(tmp_path, fail_times=0, variant=variant, os_name=os_name)
    assert "RESULT=ok" in r.stdout, r.stdout + r.stderr
    assert len(calls) == 1
    if expected:
        assert expected in calls[0]
    else:
        assert "--extra-index-url" not in calls[0]


# ── install.ps1 ─────────────────────────────────────────────────────────────
needs_pwsh = pytest.mark.skipif(PWSH is None, reason="pwsh not available")

_EXTRACT = r"""
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($env:PS1_UNDER_TEST, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw "parse errors: $($errors.Count)" }
function Import-InstallerFunction([string]$Name) {
  $fn = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $Name }, $true)
  if (-not $fn) { throw "function $Name not found" }
  . ([scriptblock]::Create($fn.Extent.Text))
  Set-Item -Path "function:global:$Name" -Value (Get-Item "function:$Name").ScriptBlock
}
"""


def _pwsh(body, env=None):
    e = _clean_env(PS1_UNDER_TEST=PS1, **(env or {}))
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", _EXTRACT + body],
                       capture_output=True, text=True, env=e, timeout=120)
    return r


@needs_pwsh
def test_ps1_models_accepts_unquoted_list_and_quoted_string():
    for models in ("omnivoice,kittentts", "'omnivoice,kittentts'", "'omnivoice, kittentts'"):
        r = _pwsh(f"""
            & $env:PS1_UNDER_TEST -Help -Models {models} | Out-Null
            Import-InstallerFunction 'Join-ModelList'
            function Probe([string[]]$Models) {{ Join-ModelList $Models }}
            Write-Output ("JOINED=" + (Probe -Models {models}))
        """)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "JOINED=omnivoice,kittentts" in r.stdout, r.stdout + r.stderr


@needs_pwsh
def test_ps1_set_default_env_keeps_user_values():
    r = _pwsh("""
        function Write-Info([string]$Message) { }
        Import-InstallerFunction 'Set-DefaultEnv'
        $env:UV_HTTP_TIMEOUT = '42'
        Set-DefaultEnv 'UV_HTTP_TIMEOUT' '600'
        Set-DefaultEnv 'UV_HTTP_RETRIES' '5'
        Write-Output "VALUES=$env:UV_HTTP_TIMEOUT,$env:UV_HTTP_RETRIES"
    """)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VALUES=42,5" in r.stdout


def _ps_repair(tmp_path, variant, fail_times, env_has_uv=True):
    return _pwsh(f"""
        $global:ApplioDir = '{tmp_path}'
        $global:Variant = '{variant}'
        $global:ApplioInstallAttempts = 3
        $global:Calls = New-Object System.Collections.Generic.List[string]
        $global:Healthy = $false
        function global:Write-Info([string]$Message) {{ }}
        function global:Write-Warn2([string]$Message) {{ }}
        function global:Start-Sleep {{ param([int]$Seconds) }}
        function global:Get-ApplioPython {{ 'C:\\applio\\env\\python.exe' }}
        function global:Invoke-Quiet([scriptblock]$Command) {{ $global:LASTEXITCODE = {0 if env_has_uv else 1} }}
        function global:Test-ApplioHealthy {{ $global:Healthy }}
        function global:Invoke-Native {{
          param([string]$FilePath, [string[]]$ArgumentList = @())
          $global:Calls.Add((@($FilePath) + $ArgumentList) -join ' ')
          if ($global:Calls.Count -le {fail_times}) {{ throw "$FilePath exited with code 2" }}
          $global:Healthy = $true
        }}
        Import-InstallerFunction 'Repair-ApplioEnvironment'
        $result = Repair-ApplioEnvironment
        Write-Output ("RESULT=" + (ConvertTo-Json -Compress @{{ ok = $result; calls = @($global:Calls) }}))
    """)


def _result(r):
    line = next(l for l in r.stdout.splitlines() if l.startswith("RESULT="))
    return json.loads(line[len("RESULT="):])


@needs_pwsh
def test_ps1_applio_repair_retries_with_env_python(tmp_path):
    r = _ps_repair(tmp_path, "cuda", fail_times=2)
    assert r.returncode == 0, r.stdout + r.stderr
    res = _result(r)
    assert res["ok"] is True
    assert len(res["calls"]) == 3
    assert res["calls"][0] == (
        "C:\\applio\\env\\python.exe -m uv pip install -r requirements.txt --python C:\\applio\\env\\python.exe "
        "--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match"
    )


@needs_pwsh
def test_ps1_applio_repair_gives_up_after_three(tmp_path):
    r = _ps_repair(tmp_path, "cuda", fail_times=99)
    assert r.returncode == 0, r.stdout + r.stderr
    res = _result(r)
    assert res["ok"] is False
    assert len(res["calls"]) == 3


@needs_pwsh
def test_ps1_applio_repair_cpu_index_and_path_uv_fallback(tmp_path):
    r = _ps_repair(tmp_path, "cpu", fail_times=0, env_has_uv=False)
    assert r.returncode == 0, r.stdout + r.stderr
    res = _result(r)
    assert res["ok"] is True
    assert res["calls"] == [
        "uv pip install -r requirements.txt --python C:\\applio\\env\\python.exe "
        "--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
    ]


def test_ps1_static_contract():
    src = open(PS1, encoding="utf-8").read()
    # Network defaults applied before setup:api and Applio, user values kept.
    for key, value in (("UV_HTTP_TIMEOUT", "600"), ("UV_HTTP_RETRIES", "5"), ("OMNIVOICE_CUDNN8_TIMEOUT", "1800")):
        assert f"'{key}'" in src and f"'{value}'" in src
    assert src.index("Set-DefaultEnv $k") < src.index("'run', 'setup:api'")
    # Bun's installer runs in a child PowerShell, not as a scriptblock in this session.
    assert "[scriptblock]::Create((Invoke-RestMethod 'https://bun.sh/install.ps1'))" not in src
    assert "-File $installerPath -Version $Version" in src
    # Native calls relax ErrorActionPreference and judge by exit code.
    native = src[src.index("function Invoke-Native"):src.index("function Set-DefaultEnv")]
    assert "$ErrorActionPreference = 'Continue'" in native and "$code -ne 0" in native
    # -Models is an array parameter.
    assert "[string[]]$Models" in src


def test_setup_py_cudnn8_timeout_env():
    import importlib.util
    spec = importlib.util.spec_from_file_location("vs_setup", os.path.join(REPO, "scripts", "setup.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = os.environ.pop("OMNIVOICE_CUDNN8_TIMEOUT", None)
    try:
        assert mod._cudnn8_timeout() == 180
        os.environ["OMNIVOICE_CUDNN8_TIMEOUT"] = "1800"
        assert mod._cudnn8_timeout() == 1800
        os.environ["OMNIVOICE_CUDNN8_TIMEOUT"] = "junk"
        assert mod._cudnn8_timeout() == 180
    finally:
        os.environ.pop("OMNIVOICE_CUDNN8_TIMEOUT", None)
        if old is not None:
            os.environ["OMNIVOICE_CUDNN8_TIMEOUT"] = old
