param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [string]$Username = "root",

    [string]$Identity = "$env:USERPROFILE\.ssh\id_ed25519",

    [string]$DicomBundlePath = "",

    [string]$SmriZipPath = "C:\Users\Devab\Downloads\SMRI.zip",

    [string]$RemoteBaseDir = "/root/mm_dbgdgm_inputs"
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($DicomBundlePath)) {
    $DicomBundlePath = Join-Path $PSScriptRoot 'modal_training_artifacts\dicom_bundle\prepared_dicom_bundle.zip'
}

$DicomBundlePath = (Resolve-Path -Path $DicomBundlePath).Path
$SmriZipPath = (Resolve-Path -Path $SmriZipPath).Path

if (-not (Test-Path $DicomBundlePath)) {
    throw "DICOM bundle not found: $DicomBundlePath"
}

if (-not (Test-Path $SmriZipPath)) {
    throw "SMRI zip not found: $SmriZipPath"
}

Write-Host "Uploading to $Username@$HostName"
Write-Host "DICOM bundle: $DicomBundlePath"
Write-Host "SMRI zip: $SmriZipPath"
Write-Host "Remote base dir: $RemoteBaseDir"

$sshArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=30'
)
if (-not [string]::IsNullOrWhiteSpace($Identity)) {
    $sshArgs += @('-i', $Identity)
}
$sshArgs += @("$Username@$HostName", "mkdir -p '$RemoteBaseDir'")

Write-Host "Ensuring remote directory exists..."
& ssh @sshArgs

if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directory on $HostName"
}

function Start-ScpJob {
    param(
        [string]$JobName,
        [string]$SourcePath,
        [string]$RemotePath,
        [string]$HostName,
        [string]$UserName,
        [string]$IdentityPath
    )

    Start-Job -Name $JobName -ScriptBlock {
        param($SourcePath, $RemotePath, $HostName, $UserName, $IdentityPath)

        $scpArgs = @(
            '-C',
            '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=30'
        )
        if (-not [string]::IsNullOrWhiteSpace($IdentityPath)) {
            $scpArgs += @('-i', $IdentityPath)
        }
        $scpTarget = '{0}@{1}:{2}' -f $UserName, $HostName, $RemotePath
        $scpArgs += @($SourcePath, $scpTarget)

        & scp @scpArgs
        exit $LASTEXITCODE
    } -ArgumentList $SourcePath, $RemotePath, $HostName, $UserName, $IdentityPath
}

$jobs = @()
$jobs += Start-ScpJob -JobName 'dicom-upload' -SourcePath $DicomBundlePath -RemotePath "$RemoteBaseDir/$(Split-Path $DicomBundlePath -Leaf)" -HostName $HostName -UserName $Username -IdentityPath $Identity
$jobs += Start-ScpJob -JobName 'smri-upload' -SourcePath $SmriZipPath -RemotePath "$RemoteBaseDir/$(Split-Path $SmriZipPath -Leaf)" -HostName $HostName -UserName $Username -IdentityPath $Identity

Write-Host "Waiting for parallel uploads to finish..."
Wait-Job -Job $jobs | Out-Null

$failed = $false
foreach ($job in $jobs) {
    $output = Receive-Job -Job $job -ErrorAction SilentlyContinue
    if ($job.State -ne 'Completed') {
        $failed = $true
        Write-Host "[$($job.Name)] failed with state: $($job.State)"
        if ($output) {
            Write-Host $output
        }
    } else {
        Write-Host "[$($job.Name)] completed"
    }
    Remove-Job -Job $job | Out-Null
}

if ($failed) {
    throw 'One or more uploads failed.'
}

Write-Host "Upload completed successfully."
Write-Host "Remote files:"
Write-Host "  $RemoteBaseDir/$(Split-Path $DicomBundlePath -Leaf)"
Write-Host "  $RemoteBaseDir/$(Split-Path $SmriZipPath -Leaf)"
