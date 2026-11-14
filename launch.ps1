param(
    [ValidateRange(1024, 65535)][int]$Port = 7860,
    [switch]$Rebuild,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:VOCAB_RAG_PORT = [string]$Port
$env:NO_PROXY = "127.0.0.1,localhost,$env:NO_PROXY".TrimEnd(',')
$url = "http://127.0.0.1:$Port/"
$server = $null
$startupLock = $null

function Test-VocabApp {
    try {
        $request = [Net.WebRequest]::Create($url + 'config')
        $request.Proxy = $null
        $request.Timeout = 1000
        $request.ReadWriteTimeout = 1000
        $response = $request.GetResponse()
        try {
            $reader = [IO.StreamReader]::new($response.GetResponseStream())
            # Windows PowerShell cannot parse empty keys elsewhere in Gradio's config.
            $title = [regex]::Match($reader.ReadToEnd(), '"title"\s*:\s*"英语词义分类 RAG 知识库（原型）"')
            return $title.Success
        } finally {
            $response.Dispose()
        }
    } catch {
        return $false
    }
}

function Open-VocabApp {
    Write-Host "知识库已就绪 $url"
    if (-not $NoBrowser) {
        try { Start-Process $url } catch { Write-Warning '请手动在浏览器中打开上面的地址。' }
    }
}

function Invoke-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python 命令失败（退出码 $LASTEXITCODE）。" }
}

try {
    New-Item -ItemType Directory -Path 'data' -Force | Out-Null
    try {
        $startupLock = [IO.File]::Open((Join-Path $PSScriptRoot 'data\.launcher.lock'),
            'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.IOException] {
        throw '另一个启动器正在准备知识库，请等它完成。'
    }

    if (Test-VocabApp) {
        if ($Rebuild) { throw '请先停止正在运行的知识库，再重建索引。' }
        Open-VocabApp
        return
    }
    $listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if ($listeners.Port -contains $Port) { throw "端口 $Port 已被占用，请用 -Port 指定另一个端口。" }

    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host '正在创建项目的 Python 环境……'
        $systemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        $compatiblePython = $false
        if ($systemPython) {
            & $systemPython.Source -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'
            $compatiblePython = $LASTEXITCODE -eq 0
        }
        if ($compatiblePython) {
            & $systemPython.Source -m venv .venv
        } elseif ($pyLauncher) {
            & $pyLauncher.Source -3.12 -m venv .venv
        } else {
            throw '请先安装 Python 3.12，并将它加入 PATH。'
        }
        if ($LASTEXITCODE -ne 0) { throw '创建 Python 环境失败，请确认 Python 3.12 已安装。' }
    }

    $requirementsHash = Invoke-Python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('requirements.txt').read_bytes()).hexdigest())"
    $installedHash = Get-Content -LiteralPath '.venv\.requirements.sha256' -ErrorAction SilentlyContinue
    if ($requirementsHash -ne $installedHash) {
        Write-Host '正在安装项目依赖……'
        Invoke-Python -m ensurepip --upgrade
        Invoke-Python -m pip install -r requirements.txt
        Set-Content -LiteralPath '.venv\.requirements.sha256' -Value $requirementsHash -Encoding ascii
    }

    if (-not $env:FASTEMBED_CACHE_PATH) {
        $env:FASTEMBED_CACHE_PATH = Join-Path $PSScriptRoot 'data\model-cache'
    }
    if ($Rebuild -or -not (Test-Path -LiteralPath 'data\index.zip' -PathType Leaf)) {
        if (-not (Test-Path -LiteralPath 'data\vault')) {
            Write-Host '正在复制随附的示例词库……'
            Copy-Item -LiteralPath 'examples\vault' -Destination 'data\vault' -Recurse
        }
        Write-Host '正在构建索引。首次使用可能需要下载词向量模型……'
        Invoke-Python -B -u parse_vault.py
        Invoke-Python -B -u build_index.py
    }

    Write-Host '正在启动本地网页……'
    $logFile = Join-Path $PSScriptRoot "data\launcher-$Port.log"
    $errorLog = Join-Path $PSScriptRoot "data\launcher-$Port-error.log"
    $launch = @{
        FilePath = $python
        ArgumentList = '-B', '-u', 'app.py'
        WindowStyle = 'Hidden'
        RedirectStandardOutput = $logFile
        RedirectStandardError = $errorLog
        PassThru = $true
    }
    $server = Start-Process @launch

    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    while (-not (Test-VocabApp)) {
        if ($server.HasExited) { throw "网页启动失败，请查看 $errorLog。" }
        if ([DateTime]::UtcNow -gt $deadline) { throw "网页启动超时，请查看 $errorLog。" }
        Start-Sleep -Milliseconds 300
    }
    if ($server.HasExited) { throw "网页进程已退出，请查看 $errorLog。" }
    $startupLock.Dispose()
    $startupLock = $null
    Open-VocabApp
    Write-Host '请保持窗口开启。按 Ctrl+C 停止。'
    while (-not $server.HasExited) { Start-Sleep -Milliseconds 500 }
    if ($server.ExitCode -ne 0) { throw "网页进程异常退出，请查看 $errorLog。" }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -ErrorAction SilentlyContinue }
    if ($startupLock) { $startupLock.Dispose() }
}
