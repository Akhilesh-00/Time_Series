# Shows progress of LITE ablation runs.
# Usage:  .\scripts\check_status.ps1            (snapshot of all experiments)
#         .\scripts\check_status.ps1 -Follow    (live tail of the newest log, Ctrl+C to stop)
param([switch]$Follow)
Set-Location (Join-Path $PSScriptRoot "..")

$running = Get-Process python -ErrorAction SilentlyContinue
if ($running) { Write-Host "Status: RUNNING (python PID $($running.Id -join ', '))" -ForegroundColor Green }
else          { Write-Host "Status: NOT RUNNING" -ForegroundColor Yellow }

foreach ($csv in Get-ChildItem results_ablation -Recurse -Filter results_per_run.csv -ErrorAction SilentlyContinue) {
    $rows = @(Import-Csv $csv.FullName)
    Write-Host "`n== $($csv.Directory.Name): $($rows.Count) runs done ==" -ForegroundColor Cyan
    $rows | Format-Table dataset, condition, seed,
        @{n='accuracy';e={[math]::Round([double]$_.accuracy,4)}},
        @{n='train_s';e={[math]::Round([double]$_.training_seconds,1)}},
        epochs_run, best_epoch -AutoSize | Out-String | Write-Host
}

$log = Get-ChildItem results_ablation_*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) {
    $current = Get-ChildItem results_ablation -Recurse -Directory -Filter "seed_*" -ErrorAction SilentlyContinue |
        Sort-Object CreationTime | Select-Object -Last 1
    if ($current) { Write-Host "Current run: $($current.Parent.Parent.Name) / $($current.Parent.Name) / $($current.Name)" }
    $epoch = Select-String -Path $log.FullName -Pattern "^Epoch \d+/\d+" | Select-Object -Last 1
    if ($epoch) { Write-Host "Last epoch: $($epoch.Line)" }
    Write-Host "`n--- $($log.Name) (last lines) ---"
    if ($Follow) { Get-Content $log.FullName -Tail 10 -Wait } else { Get-Content $log.FullName -Tail 6 }
}
