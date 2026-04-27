[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [string]$Username = "root",

    [string]$Identity = "$env:USERPROFILE\.ssh\id_ed25519",

    [string]$FmriZipPath = "C:\Users\Devab\Downloads\FMRI_DOWNLOAD_dataset.zip",

    [string]$SmriZipPath = "C:\Users\Devab\Downloads\SMRI DOWNLOAD_dataset.zip",

    [string]$RemoteBaseDir = "/root/mm_dbgdgm_inputs",

    [string]$LogDir = ""
)

$ErrorActionPreference = 'Stop'

$scriptPath = Join-Path $PSScriptRoot 'upload_vm_inputs.ps1'
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Required helper script not found: $scriptPath"
}

if (-not (Test-Path -LiteralPath $FmriZipPath)) {
    throw "fMRI ZIP not found: $FmriZipPath"
}

if (-not (Test-Path -LiteralPath $SmriZipPath)) {
    throw "sMRI ZIP not found: $SmriZipPath"
}

$resolvedFmriZip = (Resolve-Path -LiteralPath $FmriZipPath).Path
$resolvedSmriZip = (Resolve-Path -LiteralPath $SmriZipPath).Path

Write-Host "Starting VM upload with local datasets..."
Write-Host "  Host: $Username@$HostName"
Write-Host "  fMRI: $resolvedFmriZip"
Write-Host "  sMRI: $resolvedSmriZip"
Write-Host "  Remote directory: $RemoteBaseDir"

$invokeArgs = @{
    HostName = $HostName
    Username = $Username
    Identity = $Identity
    DicomBundlePath = $resolvedFmriZip
    SmriZipPath = $resolvedSmriZip
    RemoteBaseDir = $RemoteBaseDir
}

if (-not [string]::IsNullOrWhiteSpace($LogDir)) {
    $invokeArgs['LogDir'] = $LogDir
}

& $scriptPath @invokeArgs

