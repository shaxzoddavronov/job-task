# Start db + api + bot with Docker on Windows. Uses the NVIDIA GPU automatically when Docker
# Desktop can reach one (WSL2 backend + NVIDIA driver), otherwise runs on CPU.
# Extra arguments go to `docker compose up`, e.g.  .\start.ps1 -d
Set-Location -Path $PSScriptRoot

$files = @("-f", "docker-compose.yml")
docker run --rm --gpus all busybox:1.36 true *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "NVIDIA GPU available to Docker: starting in GPU mode (BGE-M3 + reranker)."
    $files += @("-f", "docker-compose.gpu.yml")
} else {
    Write-Host "No GPU available to Docker: starting in CPU mode (BGE-M3 only, slower)."
}

# Bot token: BOT_TOKEN from the environment or .env, else bot_token.txt (never printed)
$envHasToken = (Test-Path .env) -and (Select-String -Path .env -Pattern '^BOT_TOKEN=.' -Quiet)
if (-not $env:BOT_TOKEN -and -not $envHasToken -and (Test-Path bot_token.txt)) {
    $env:BOT_TOKEN = (Get-Content bot_token.txt -Raw).Trim()
}

docker compose @files up --build @args
