param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DropletIp,

    [string]$Identity,

    [string]$Destination = (Join-Path (Split-Path -Parent $PSScriptRoot) 'local_results'),

    [string]$RemoteUser = 'root',

    [string]$RemoteResults = '/mnt/trainingresults',

    [switch]$All
)

$ErrorActionPreference = 'Stop'

if ($Identity -and -not (Test-Path -LiteralPath $Identity -PathType Leaf)) {
    throw "Identity file does not exist: $Identity"
}

$sshCommand = Get-Command ssh.exe -ErrorAction SilentlyContinue
$scpCommand = Get-Command scp.exe -ErrorAction SilentlyContinue
$tarCommand = Get-Command tar.exe -ErrorAction SilentlyContinue

if (-not $sshCommand) {
    throw 'ssh.exe was not found. Install the Windows OpenSSH Client feature.'
}

if (-not $scpCommand) {
    throw 'scp.exe was not found. Install the Windows OpenSSH Client feature.'
}

if (-not $tarCommand) {
    throw 'tar.exe was not found. Install the Windows bsdtar feature or Git for Windows.'
}

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

$remoteBase = "{0}@{1}" -f $RemoteUser, $DropletIp
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$remoteArchive = "/tmp/trainingresults_${timestamp}.tar"
$localArchive = Join-Path $env:TEMP ("trainingresults_{0}.tar" -f $timestamp)

$sshArgs = @()
$scpArgs = @()
if ($Identity) {
    $sshArgs += @('-i', $Identity)
    $scpArgs += @('-i', $Identity)
}

Write-Host ''
Write-Host '============================================================'
Write-Host '  Downloading results from DigitalOcean Droplet'
Write-Host "  Droplet IP  : $DropletIp"
Write-Host "  Remote path : $RemoteResults"
Write-Host "  Local target: $destinationPath"
if ($All) {
    Write-Host '  Content     : full results volume'
} else {
    Write-Host '  Content     : logs + model weights only'
}
Write-Host '============================================================'
Write-Host ''

$remoteTarCommand = if ($All) {
    "test -d '{0}' && tar -cf '{1}' -C '{0}' ." -f $RemoteResults, $remoteArchive
} else {
    @(
        ("cd '{0}'" -f $RemoteResults)
        'found=0'
        'for path in logs models_*; do if [ -e "$path" ]; then found=1; break; fi; done'
        'if [ "$found" -ne 1 ]; then echo ''No logs or models found to archive.'' >&2; exit 1; fi'
        ("tar -cf '{0}' --ignore-failed-read logs models_*" -f $remoteArchive)
    ) -join '; '
}
Write-Host "[1] Creating remote archive: $remoteArchive"
& $sshCommand.Source @sshArgs $remoteBase $remoteTarCommand
if ($LASTEXITCODE -ne 0) {
    throw 'ssh.exe failed while creating the remote archive.'
}

try {
    Write-Host "[2] Downloading archive to: $localArchive"
    & $scpCommand.Source @scpArgs ("{0}:{1}" -f $remoteBase, $remoteArchive) $localArchive
    if ($LASTEXITCODE -ne 0) {
        throw 'scp.exe failed while downloading the results archive.'
    }

    Write-Host "[3] Extracting archive into: $destinationPath"
    & $tarCommand.Source -xf $localArchive -C $destinationPath
    if ($LASTEXITCODE -ne 0) {
        throw 'tar.exe failed while extracting the results archive.'
    }
}
finally {
    Write-Host "[4] Cleaning up remote archive: $remoteArchive"
    & $sshCommand.Source @sshArgs $remoteBase "rm -f '$remoteArchive'"
    if (Test-Path -LiteralPath $localArchive -PathType Leaf) {
        Remove-Item -LiteralPath $localArchive -Force
    }
}

Write-Host ''
Write-Host '============================================================'
Write-Host '  Sync complete'
Write-Host "  Local results: $destinationPath"
Write-Host '============================================================'
Write-Host ''