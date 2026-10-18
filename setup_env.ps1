# ============================================================
#  图片匹配音乐 Demo - 依赖安装脚本
#
#  做四件事：
#    1. 找到可用的 Python
#    2. 创建项目虚拟环境 .venv
#    3. 安装 CPU 版 torch
#    4. 安装其余依赖（Flask / Pillow / numpy / transformers 等）
#
#  用法：
#     .\setup_env.ps1
#
#  若提示"禁止运行脚本"，先执行一次：
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
#  注意：这一步需要联网，且会下载约 200 MB 的 torch。
#        之后的 Web 演示阶段完全离线。
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# ---- 1. 找 Python ----
$dsh = Join-Path $env:LOCALAPPDATA `
    "Programs\DeepSeek Harness\resources\runtime\primary-runtime\dependencies\python\python.exe"

$basePy = $null
if (Test-Path -LiteralPath $dsh) {
    $basePy = $dsh
    Write-Host "[信息] 使用 DSH 内置解释器创建虚拟环境"
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        $basePy = $cmd.Source
        Write-Host "[信息] 使用 PATH 中的 python：$basePy"
    }
}

if (-not $basePy) {
    Write-Host ""
    Write-Host "[错误] 没有找到可用的 Python。" -ForegroundColor Red
    Write-Host "请先安装：winget install Python.Python.3.12"
    Write-Host ""
    exit 1
}

$ver = & $basePy -c "import sys; print(sys.version.split()[0])"
Write-Host "[信息] 基础解释器版本：Python $ver"

# ---- 2. 创建虚拟环境 ----
# 注意：不能只看 .venv\Scripts\python.exe 是否存在。
# 虚拟环境是"绑定创建时那台机器"的：把项目从别的电脑拷过来后，
# python.exe 还在，但 pyvenv.cfg 里的 home 指向一个本机不存在的路径，
# 于是这个解释器一启动就失败——后面的 pip 会报出与真实原因无关的
# "torch 安装失败"，让人去查网络和磁盘。
# 因此这里必须**实际运行一次**来确认它可用，不可用就重建。
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvOk = $false

if (Test-Path -LiteralPath $venvPy) {
    & $venvPy -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $venvOk = $true
        Write-Host "[信息] 虚拟环境已存在且可用，跳过创建"
    } else {
        $cfg = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
        Write-Host "[警告] 已存在的 .venv 无法启动，将重建。" -ForegroundColor Yellow
        if (Test-Path -LiteralPath $cfg) {
            $homeLine = (Select-String -LiteralPath $cfg -Pattern '^home\s*=' |
                         Select-Object -First 1).Line
            if ($homeLine) {
                Write-Host "         它的基础解释器指向：$homeLine"
            }
            Write-Host "         （常见原因：项目从另一台电脑复制而来）"
        }
        Remove-Item -LiteralPath (Join-Path $PSScriptRoot ".venv") -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $venvPy) {
            Write-Host "[错误] 无法删除旧的 .venv（可能有进程正在占用）。" -ForegroundColor Red
            Write-Host "       请关闭正在使用该虚拟环境的终端后重试，或手动删除 .venv 目录。"
            exit 1
        }
    }
}

if (-not $venvOk) {
    Write-Host "[信息] 正在创建虚拟环境 .venv ..."
    & $basePy -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 创建虚拟环境失败" -ForegroundColor Red
        exit 1
    }
    & $venvPy -c "import sys"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 新建的虚拟环境仍无法启动，请检查基础解释器：$basePy" -ForegroundColor Red
        exit 1
    }
}

# ---- 沙箱友好的临时目录（避免写系统临时目录被拒）----
$tmpDir = Join-Path $PSScriptRoot ".tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$env:TMP = $tmpDir
$env:TEMP = $tmpDir

# ---- 3. 安装 CPU 版 torch ----
Write-Host ""
Write-Host "[信息] 安装 CPU 版 torch（约 200 MB，需要联网）..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install torch==2.4.1 `
    --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] torch 安装失败。若为磁盘空间或网络问题，请重试。" -ForegroundColor Red
    exit 1
}

# ---- 4. 安装其余依赖 ----
Write-Host ""
Write-Host "[信息] 安装其余依赖 ..." -ForegroundColor Cyan
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] 依赖安装失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== 环境准备完成 ===" -ForegroundColor Green
Write-Host ""
Write-Host "接下来："
Write-Host "  1. .\run_selfcheck.ps1        静态自检"
Write-Host "  2. 下载 CLIP 模型缓存（见 README 第 3 步）"
Write-Host "  3. .\.venv\Scripts\python.exe clip_encoder.py   模型契约自检"
Write-Host "  4. .\.venv\Scripts\python.exe -m pytest -v      运行测试"
Write-Host ""
exit 0
