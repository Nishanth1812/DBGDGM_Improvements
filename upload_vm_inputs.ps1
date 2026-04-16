param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [string]$Username = "root",

    [string]$Identity = "$env:USERPROFILE\.ssh\id_ed25519",

    [string]$DicomBundlePath = "",

    [string]$SmriZipPath = "C:\Users\Devab\Downloads\SMRI.zip",

    [string]$RemoteBaseDir = "/root/mm_dbgdgm_inputs",

    [string]$LogDir = ""
)

$ErrorActionPreference = 'Stop'

function Format-Bytes {
    param([Int64]$Bytes)

    $units = @('B', 'KB', 'MB', 'GB', 'TB')
    $value = [double]$Bytes
    $unitIndex = 0

    while ($value -ge 1024 -and $unitIndex -lt ($units.Count - 1)) {
        $value /= 1024
        $unitIndex++
    }

    if ($unitIndex -eq 0) {
        return "{0} {1}" -f [int64]$value, $units[$unitIndex]
    }

    return "{0:N2} {1}" -f $value, $units[$unitIndex]
}

function Write-Log {
    param([string]$Message)

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$timestamp] $Message"
    Write-Host $line
    if ($script:MasterLogPath) {
        Add-Content -LiteralPath $script:MasterLogPath -Value $line
    }
}

function Convert-ScpSizeToBytes {
    param([string]$Value)

    $cleanValue = ($Value -replace '\s+', '').Trim()
    if ([string]::IsNullOrWhiteSpace($cleanValue)) {
        return 0L
    }

    if ($cleanValue -match '^(?<number>\d+(?:\.\d+)?)(?<unit>[KMGTP]?B)$') {
        $number = [double]$Matches.number
        $unit = $Matches.unit.ToUpperInvariant()
        switch ($unit) {
            'B' { return [Int64][math]::Round($number) }
            'KB' { return [Int64][math]::Round($number * 1KB) }
            'MB' { return [Int64][math]::Round($number * 1MB) }
            'GB' { return [Int64][math]::Round($number * 1GB) }
            'TB' { return [Int64][math]::Round($number * 1TB) }
            'PB' { return [Int64][math]::Round($number * 1PB) }
        }
    }

    return 0L
}

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $PSScriptRoot 'logs\upload_vm_inputs'
}

$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogDir = Join-Path $LogDir $runStamp
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$script:MasterLogPath = Join-Path $LogDir 'upload.log'

if ([string]::IsNullOrWhiteSpace($DicomBundlePath)) {
    $DicomBundlePath = 'C:\Users\Devab\Downloads\dicom_bundle\prepared_dicom_bundle.zip'
}

if (-not (Test-Path -LiteralPath $DicomBundlePath)) {
    throw "DICOM bundle not found: $DicomBundlePath"
}

if (-not (Test-Path -LiteralPath $SmriZipPath)) {
    throw "SMRI zip not found: $SmriZipPath"
}

$DicomBundlePath = (Resolve-Path -LiteralPath $DicomBundlePath).Path
$SmriZipPath = (Resolve-Path -LiteralPath $SmriZipPath).Path

Write-Log "Uploading to $Username@$HostName"
Write-Log "Log directory: $LogDir"
Write-Log "Master log: $script:MasterLogPath"
Write-Log "DICOM bundle: $DicomBundlePath"
Write-Log "DICOM size: $(Format-Bytes ((Get-Item -LiteralPath $DicomBundlePath).Length))"
Write-Log "SMRI zip: $SmriZipPath"
Write-Log "SMRI size: $(Format-Bytes ((Get-Item -LiteralPath $SmriZipPath).Length))"
Write-Log "Remote base dir: $RemoteBaseDir"

$sshArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=30'
)
if (-not [string]::IsNullOrWhiteSpace($Identity)) {
    $sshArgs += @('-i', $Identity)
}
$sshArgs += @("$Username@$HostName", "mkdir -p '$RemoteBaseDir'")

Write-Log "Ensuring remote directory exists..."
& ssh @sshArgs

if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directory on $HostName"
}

Write-Log "Remote directory ready: $RemoteBaseDir"

function Start-ScpJob {
    param(
        [string]$JobName,
        [string]$SourcePath,
        [string]$RemotePath,
        [string]$HostName,
        [string]$UserName,
        [string]$IdentityPath,
        [string]$LogPath,
        [string]$SourceSizeText
    )

    Start-Job -Name $JobName -ScriptBlock {
        param($SourcePath, $RemotePath, $HostName, $UserName, $IdentityPath, $LogPath, $SourceSizeText, $JobName)

        function Emit-Event {
            param(
                [string]$Type,
                [string]$Message,
                [int]$Percent = 0,
                [string]$TransferredText = '',
                [string]$RateText = '',
                [string]$EtaText = ''
            )

            $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            $line = "[$timestamp][$JobName] $Message"
            Add-Content -LiteralPath $LogPath -Value $line
            Write-Output ([pscustomobject]@{
                Type = $Type
                JobName = $JobName
                Timestamp = $timestamp
                Message = $Message
                Percent = $Percent
                TransferredText = $TransferredText
                RateText = $RateText
                EtaText = $EtaText
            })
        }

        $scpArgs = @(
            '-C',
            '-o', 'LogLevel=ERROR',
            '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=30'
        )
        if (-not [string]::IsNullOrWhiteSpace($IdentityPath)) {
            $scpArgs += @('-i', $IdentityPath)
        }
        $scpTarget = '{0}@{1}:{2}' -f $UserName, $HostName, $RemotePath
        $scpArgs += @($SourcePath, $scpTarget)

        Emit-Event -Type 'Start' -Message ("Starting upload of {0} ({1}) to {2}" -f (Split-Path $SourcePath -Leaf), $SourceSizeText, $scpTarget)

        $progressPattern = '^\s*(?<label>.*?)\s+(?<percent>\d{1,3})%\s+(?<transferred>[0-9.]+[KMGTP]?B)\s+(?<rate>[0-9.]+[KMGTP]?B/s)\s+(?<eta>.+)$'

        & scp @scpArgs 2>&1 | ForEach-Object {
            $text = $_.ToString().TrimEnd()
            if ([string]::IsNullOrWhiteSpace($text)) {
                return
            }

            if ($text -match $progressPattern) {
                Emit-Event -Type 'Progress' -Message 'Progress update' -Percent ([int]$Matches.percent) -TransferredText $Matches.transferred -RateText $Matches.rate -EtaText $Matches.eta
                return
            }

            if ($text -match '^(debug\d+:|Authenticated to |Sending subsystem: |Entering interactive session\.|Requesting no-more-sessions@openssh\.com|SSH2_MSG_|channel \d+:|pledge: |Remote: |Server accepts key: |debug1: |Transferred: )') {
                return
            }

            Emit-Event -Type 'Message' -Message $text
        }

        $scpExitCode = $LASTEXITCODE
        if ($scpExitCode -ne 0) {
            Emit-Event -Type 'Failure' -Message "Upload failed with exit code $scpExitCode"
            throw "scp exited with code $scpExitCode"
        }

        Emit-Event -Type 'Completed' -Message "Upload completed successfully"
    } -ArgumentList $SourcePath, $RemotePath, $HostName, $UserName, $IdentityPath, $LogPath, $SourceSizeText, $JobName
}

Write-Log "Starting parallel uploads..."

$jobInfos = @(
    [pscustomobject]@{
        Name = 'dicom-upload'
        LogPath = Join-Path $LogDir 'dicom-upload.log'
        RemotePath = "$RemoteBaseDir/$(Split-Path $DicomBundlePath -Leaf)"
        Job = Start-ScpJob -JobName 'dicom-upload' -SourcePath $DicomBundlePath -RemotePath "$RemoteBaseDir/$(Split-Path $DicomBundlePath -Leaf)" -HostName $HostName -UserName $Username -IdentityPath $Identity -LogPath (Join-Path $LogDir 'dicom-upload.log') -SourceSizeText (Format-Bytes ((Get-Item -LiteralPath $DicomBundlePath).Length))
    },
    [pscustomobject]@{
        Name = 'smri-upload'
        LogPath = Join-Path $LogDir 'smri-upload.log'
        RemotePath = "$RemoteBaseDir/$(Split-Path $SmriZipPath -Leaf)"
        Job = Start-ScpJob -JobName 'smri-upload' -SourcePath $SmriZipPath -RemotePath "$RemoteBaseDir/$(Split-Path $SmriZipPath -Leaf)" -HostName $HostName -UserName $Username -IdentityPath $Identity -LogPath (Join-Path $LogDir 'smri-upload.log') -SourceSizeText (Format-Bytes ((Get-Item -LiteralPath $SmriZipPath).Length))
    }
)

Write-Log "Waiting for parallel uploads to finish..."

foreach ($jobInfo in $jobInfos) {
    $jobInfo | Add-Member -NotePropertyName LastState -NotePropertyValue $jobInfo.Job.State -Force
    $jobInfo | Add-Member -NotePropertyName Percent -NotePropertyValue 0 -Force
    $jobInfo | Add-Member -NotePropertyName TransferredBytes -NotePropertyValue 0L -Force
    $jobInfo | Add-Member -NotePropertyName RateText -NotePropertyValue '' -Force
    $jobInfo | Add-Member -NotePropertyName EtaText -NotePropertyValue '' -Force
    $jobInfo | Add-Member -NotePropertyName LastProgressPercent -NotePropertyValue -1 -Force
}

while (@($jobInfos | Where-Object { $_.Job.State -eq 'Running' -or $_.Job.State -eq 'NotStarted' }).Count -gt 0) {
    $jobOutputs = @()
    foreach ($jobInfo in $jobInfos) {
        $jobOutputs += @(Receive-Job -Job $jobInfo.Job -ErrorAction SilentlyContinue)
    }

    foreach ($event in $jobOutputs) {
        if ($null -eq $event -or -not ($event.PSObject.Properties.Name -contains 'Type')) {
            continue
        }

        $targetJob = $jobInfos | Where-Object { $_.Name -eq $event.JobName } | Select-Object -First 1
        if ($null -eq $targetJob) {
            continue
        }

        switch ($event.Type) {
            'Start' {
                Write-Log "[$($targetJob.Name)] upload started"
            }
            'Progress' {
                $targetJob.Percent = [int]$event.Percent
                $targetJob.TransferredBytes = [Int64]((($targetJob.SourceBytes) * [double]$targetJob.Percent) / 100.0)
                $targetJob.RateText = [string]$event.RateText
                $targetJob.EtaText = [string]$event.EtaText
                if ($targetJob.Percent -ne $targetJob.LastProgressPercent) {
                    $targetJob.LastProgressPercent = $targetJob.Percent
                }
            }
            'Completed' {
                $targetJob.Percent = 100
                $targetJob.TransferredBytes = [Int64]$targetJob.SourceBytes
                $targetJob.RateText = ''
                $targetJob.EtaText = 'done'
                Write-Log "[$($targetJob.Name)] upload completed"
            }
            'Failure' {
                Write-Log "[$($targetJob.Name)] $($event.Message)"
            }
            'Message' {
                Write-Log "[$($targetJob.Name)] $($event.Message)"
            }
        }
    }

    $completedCount = @($jobInfos | Where-Object { $_.Job.State -eq 'Completed' }).Count
    $failedCount = @($jobInfos | Where-Object { $_.Job.State -eq 'Failed' -or $_.Job.State -eq 'Stopped' }).Count
    $totalCount = $jobInfos.Count
    $percentComplete = if ($totalCount -gt 0) { [math]::Round(($completedCount / [double]$totalCount) * 100, 0) } else { 0 }

    foreach ($jobInfo in $jobInfos) {
        if ($jobInfo.Job.State -ne $jobInfo.LastState) {
            $jobInfo.LastState = $jobInfo.Job.State
            if ($jobInfo.Job.State -eq 'Running') {
                Write-Log "[$($jobInfo.Name)] upload started"
            } elseif ($jobInfo.Job.State -eq 'Completed') {
                Write-Log "[$($jobInfo.Name)] upload completed"
            } elseif ($jobInfo.Job.State -eq 'Failed' -or $jobInfo.Job.State -eq 'Stopped') {
                Write-Log "[$($jobInfo.Name)] upload failed with state: $($jobInfo.Job.State)"
            }
        }
    }

    Write-Progress -Id 1 -Activity "Uploading to $HostName" -Status "$completedCount/$totalCount complete | $failedCount failed" -PercentComplete $percentComplete

    foreach ($jobInfo in $jobInfos) {
        $jobTotalText = Format-Bytes $jobInfo.SourceBytes
        $jobTransferredText = Format-Bytes $jobInfo.TransferredBytes
        $jobStatus = "${jobTransferredText} / ${jobTotalText} ($($jobInfo.Percent)%)"
        if (-not [string]::IsNullOrWhiteSpace($jobInfo.RateText)) {
            $jobStatus += " | $($jobInfo.RateText)"
        }
        if (-not [string]::IsNullOrWhiteSpace($jobInfo.EtaText)) {
            $jobStatus += " | ETA $($jobInfo.EtaText)"
        }

        $progressId = if ($jobInfo.Name -eq 'dicom-upload') { 2 } else { 3 }
        $progressState = if ($jobInfo.Job.State -eq 'Completed') { 'Completed' } else { 'Uploading' }
        Write-Progress -Id $progressId -ParentId 1 -Activity "[$($jobInfo.Name)] $progressState" -Status $jobStatus -PercentComplete $jobInfo.Percent
    }

    Start-Sleep -Seconds 1
}

$failed = $false
foreach ($jobInfo in $jobInfos) {
    $output = Receive-Job -Job $jobInfo.Job -ErrorAction SilentlyContinue

    if ($jobInfo.Job.State -ne 'Completed') {
        $failed = $true
        Write-Log "[$($jobInfo.Name)] failed with state: $($jobInfo.Job.State)"
    } else {
        Write-Log "[$($jobInfo.Name)] completed"
    }
    Remove-Job -Job $jobInfo.Job | Out-Null
}

if ($failed) {
    throw 'One or more uploads failed.'
}

Write-Log "Upload completed successfully."
Write-Log "Remote files:"
Write-Log "  $RemoteBaseDir/$(Split-Path $DicomBundlePath -Leaf)"
Write-Log "  $RemoteBaseDir/$(Split-Path $SmriZipPath -Leaf)"
Write-Log "Detailed logs saved in: $LogDir"

Write-Progress -Id 1 -Completed -Activity "Uploading to $HostName"
Write-Progress -Id 2 -Completed -Activity "[dicom-upload]"
Write-Progress -Id 3 -Completed -Activity "[smri-upload]"
