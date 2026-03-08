param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DropletIp,

    [string]$Identity,

    [string]$Source = (Join-Path $PSScriptRoot 'data'),

    [string]$RemoteUser = 'root',

    [string]$RemoteRepo = '/root/DBGDGM_Improvements',

    [switch]$KeepArchive,

    [switch]$ExtractRemote
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
    throw "Source directory does not exist: $Source"
}

if ($Identity -and -not (Test-Path -LiteralPath $Identity -PathType Leaf)) {
    throw "Identity file does not exist: $Identity"
}

$tarCommand = Get-Command tar.exe -ErrorAction SilentlyContinue
$scpCommand = Get-Command scp.exe -ErrorAction SilentlyContinue

if (-not $tarCommand) {
    throw 'tar.exe was not found. Install the Windows bsdtar feature or Git for Windows.'
}

if (-not $scpCommand) {
    throw 'scp.exe was not found. Install the Windows OpenSSH Client feature.'
}

$sourceItem = Get-Item -LiteralPath $Source
$sourceParent = $sourceItem.Parent.FullName
$sourceName = $sourceItem.Name
$archivePath = Join-Path $env:TEMP ("{0}_{1}.tar" -f $sourceName, (Get-Date -Format 'yyyyMMdd_HHmmss'))
$remoteBase = "{0}@{1}" -f $RemoteUser, $DropletIp
$remoteArchive = "/tmp/{0}.tar" -f $sourceName
$remoteParent = "{0}/Alzhiemers_Training" -f $RemoteRepo.TrimEnd('/')

Write-Host "Creating archive: $archivePath"
& $tarCommand.Source -cf $archivePath -C $sourceParent $sourceName
if ($LASTEXITCODE -ne 0) {
    throw 'tar.exe failed while creating the archive.'
}

$scpArgs = @()
if ($Identity) {
    $scpArgs += @('-i', $Identity)
}
$scpArgs += @($archivePath, ("{0}:{1}" -f $remoteBase, $remoteArchive))

Write-Host "Uploading archive with scp to ${remoteBase}:${remoteArchive}"
& $scpCommand.Source @scpArgs
if ($LASTEXITCODE -ne 0) {
    throw 'scp.exe failed while uploading the archive.'
}

if ($ExtractRemote) {
    $sshCommand = Get-Command ssh.exe -ErrorAction SilentlyContinue
    if (-not $sshCommand) {
        throw 'ssh.exe was not found. Install the Windows OpenSSH Client feature to use -ExtractRemote.'
    }

    $sshArgs = @()
    if ($Identity) {
        $sshArgs += @('-i', $Identity)
    }

    $remoteCommand = "mkdir -p '$remoteParent' && tar -xf '$remoteArchive' -C '$remoteParent' && rm -f '$remoteArchive'"
    Write-Host "Extracting archive on remote host into $remoteParent"
    & $sshCommand.Source @sshArgs $remoteBase $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw 'ssh.exe failed while extracting the archive on the remote host.'
    }
} else {
    Write-Host ''
    Write-Host 'Archive upload complete.'
    Write-Host 'Extract it on the droplet with:'
    if ($Identity) {
        Write-Host "  ssh -i $Identity $remoteBase \"mkdir -p '$remoteParent' && tar -xf '$remoteArchive' -C '$remoteParent' && rm -f '$remoteArchive'\""
    } else {
        Write-Host "  ssh $remoteBase \"mkdir -p '$remoteParent' && tar -xf '$remoteArchive' -C '$remoteParent' && rm -f '$remoteArchive'\""
    }
}

if (-not $KeepArchive -and (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    Remove-Item -LiteralPath $archivePath -Force
} elseif ($KeepArchive) {
    Write-Host "Local archive kept at: $archivePath"
}

Write-Host "Upload complete: ${remoteParent}/${sourceName}"