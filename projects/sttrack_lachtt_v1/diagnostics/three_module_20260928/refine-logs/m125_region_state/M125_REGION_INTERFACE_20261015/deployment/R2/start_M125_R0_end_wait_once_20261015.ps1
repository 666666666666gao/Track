$ErrorActionPreference = 'Stop'
$taskDir = 'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007'
$pythonPath = 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe'
$reportPath = Join-Path $taskDir 'M125_REGION_DEPLOYMENT_AUDIT_R2_20261015\EXPERIMENT_AUDIT.json'
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if ($report.source_verdict -ne 'PASS_SOURCE_ONLY' -or $report.blocking_findings.Count -ne 0) {
    throw 'The source-only R2 review has not passed.'
}
$names = @('wait_M125_R0_training_end_one_shot_20261017.py', 'observe_M125_R0_training_end_R2_once_20261017.py', 'M125_R0_training_end_node_20261017.json')
$digests = @{}
foreach ($name in $names) {
    $source = Get-Item -LiteralPath (Join-Path $taskDir $name)
    $digest = (Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($report.audited_input_hashes.PSObject.Properties[$source.FullName].Value -ne ('sha256:' + $digest)) {
        throw ('The reviewed input differs: ' + $source.FullName)
    }
    $digests[$source.FullName] = $digest
}
if ([string]::IsNullOrEmpty($env:RGBD_GOAL_SSH_PASSWORD)) {
    throw 'The future SSH observer requires the authorized credential in the inherited environment.'
}
$ownerPath = Join-Path $taskDir 'M125_R0_training_end_wait_owner_20261017.json'
$launchPath = Join-Path $taskDir 'M125_R0_training_end_wait_actual_launch_20261015.json'
if ((Test-Path -LiteralPath $ownerPath) -or (Test-Path -LiteralPath $launchPath)) {
    throw 'The single wait already has an ownership or launch receipt.'
}
$helper = Join-Path $taskDir $names[0]
$waitProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-B', '-X', 'utf8', '-u', $helper) -WorkingDirectory $taskDir -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskDir 'M125_R0_end_wait_native_stdout_20261015.log') -RedirectStandardError (Join-Path $taskDir 'M125_R0_end_wait_native_stderr_20261015.log') -PassThru
$receipt = [ordered]@{
    status = 'actual_hidden_absolute_time_wait_started'
    started_at = [DateTimeOffset]::UtcNow.ToString('o')
    pid = $waitProcess.Id
    not_before = '2026-10-17T12:45:00+08:00'
    helper_sha256 = $digests[$helper]
    reviewed_input_sha256 = $digests
    R2_source_review_sha256 = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    original_R0_progress_queries = 0
    additional_local_waits = 0
    automatic_retries = 0
}
[System.IO.File]::WriteAllText($launchPath, (($receipt | ConvertTo-Json -Depth 8) + "`n"), [System.Text.UTF8Encoding]::new($false))
Start-Sleep -Seconds 1
$owner = Get-Content -LiteralPath $ownerPath -Raw | ConvertFrom-Json
if ($owner.pid -ne $waitProcess.Id -or $owner.status -ne 'one_absolute_time_wait_owned') {
    throw 'The single initial ownership receipt does not match the launched helper.'
}
$receipt | ConvertTo-Json -Depth 8 -Compress
