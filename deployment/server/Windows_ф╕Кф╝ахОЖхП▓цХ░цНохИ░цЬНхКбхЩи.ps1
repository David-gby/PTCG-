[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ServerIp,
    [string]$RemoteUser = "ubuntu",
    [string]$DesktopRuntime = "C:\Users\admin\Desktop\CardScope_Platform_v0.4.0",
    [string]$RemoteStage = "cardscope_migration"
)

$ErrorActionPreference = "Stop"
foreach ($command in "ssh", "scp") {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command was not found. Install the Windows OpenSSH Client optional feature first."
    }
}

$workspace = Join-Path $DesktopRuntime "platform_workspace"
$studioData = Join-Path $workspace "studio_data"
$uploadBatches = Join-Path $workspace "private\upload_batches"
$autoTraining = Join-Path $workspace "private\auto_training"
if (-not (Test-Path -LiteralPath $studioData -PathType Container)) {
    throw "Live desktop data was not found: $studioData"
}

$remote = "${RemoteUser}@${ServerIp}"
Write-Host "Uploading history, feedback, annotations, and the training pool directly to the server." -ForegroundColor Cyan
Write-Host "Pause enterprise uploads first and confirm at least 40 GB free disk space on the server." -ForegroundColor Yellow
ssh $remote "mkdir -p ~/$RemoteStage/platform_workspace/private"
scp -r -- $studioData "${remote}:~/$RemoteStage/platform_workspace/"
if (Test-Path -LiteralPath $uploadBatches -PathType Container) {
    scp -r -- $uploadBatches "${remote}:~/$RemoteStage/platform_workspace/private/"
}
if (Test-Path -LiteralPath $autoTraining -PathType Container) {
    scp -r -- $autoTraining "${remote}:~/$RemoteStage/platform_workspace/private/"
}

Write-Host "Bulk transfer completed. Put platform.sqlite3 and access_links.json from the private bundle in ~/$RemoteStage/platform_workspace/private/, then run finalize_migration.sh." -ForegroundColor Green
