# Diagnose (-Check) or fix (-Fix) the WSLg "[WARN:COPY MODE]" blank-window problem without wsl --shutdown.
# Cause: /mnt/shared_memory (virtiofs share between WSLg's compositor and Windows) is missing when WSL
# starts, so Weston falls back to copy mode. See https://github.com/microsoft/WSL/issues/40618.
# Works with Windows PowerShell 5.1 (no &&, no ternary). Exit codes: 0 healthy/fixed, 1 copy mode (check), 2 fix failed.
param(
  [switch]$Check,
  [switch]$Fix,
  [string]$Distro = "Ubuntu-24.04"
)
$ErrorActionPreference = "Continue"
if (-not $Check -and -not $Fix) { $Check = $true }

function Get-WslgTitles {
  $t = @(Get-Process msrdc -ErrorAction SilentlyContinue | ForEach-Object { $_.MainWindowTitle })
  return $t
}
function Get-SharedMemoryMount {
  $lines = @(wsl.exe --system -- mount 2>$null)
  return @($lines | Where-Object { $_ -match 'shared_memory' })
}
function Get-WestonPid {
  $lines = @(wsl.exe --system -- ps -eo pid,comm 2>$null)
  foreach ($l in $lines) {
    if ($l -match '^\s*(\d+)\s+weston\s*$') { return [int]$matches[1] }
  }
  return $null
}
function Get-WestonLogState {
  # returns @{ StartLine = <line no of last Weston start>; FailLine = <line no of last allocation failure>; FailText = ... }
  # two -e patterns, not "a|b": wsl.exe hands the arguments to the distro's shell, which would treat | as a pipe
  $lines = @(wsl.exe -d $Distro -- grep -n -e allocate_shared_memory -e SHARED_MEMORY_MOUNT_POINT /mnt/wslg/weston.log 2>$null)
  $start = 0; $fail = 0; $failText = ""
  foreach ($l in $lines) {
    if ($l -match '^(\d+):') {
      $n = [int]$matches[1]
      if ($l -match 'SHARED_MEMORY_MOUNT_POINT') { $start = $n }
      elseif ($l -match 'allocate_shared_memory') { $fail = $n; $failText = $l }
    }
  }
  return @{ StartLine = $start; FailLine = $fail; FailText = $failText }
}
function Show-State {
  $titles = Get-WslgTitles
  Write-Host ("WSLg windows on the Windows side: " + $(if ($titles.Count -eq 0) { "(none)" } else { ($titles | ForEach-Object { if ($_) { "'" + $_ + "'" } else { "(untitled)" } }) -join ", " }))
  $mnt = Get-SharedMemoryMount
  Write-Host ("system distro /mnt/shared_memory: " + $(if ($mnt.Count -eq 0) { "NOT mounted" } else { $mnt -join " | " }))
  $wpid = Get-WestonPid   # not $pid: that is PowerShell's read-only automatic variable
  Write-Host ("weston pid: " + $(if ($wpid) { $wpid } else { "(not running)" }))
  $st = Get-WestonLogState
  Write-Host ("weston.log: last start at line " + $st.StartLine + ", last shared-memory failure at line " + $st.FailLine + $(if ($st.FailText) { " -> " + $st.FailText } else { "" }))
  $copyTitle = @($titles | Where-Object { $_ -match 'WARN:COPY MODE' }).Count -gt 0
  $copyLog = ($st.FailLine -gt $st.StartLine)
  $virtio = @($mnt | Where-Object { $_ -match 'virtiofs' }).Count -gt 0
  return @{ CopyMode = ($copyTitle -or $copyLog); Virtio = $virtio; Pid = $wpid; State = $st }
}

if ($Check) {
  $s = Show-State
  if ($s.CopyMode) {
    Write-Host "VERDICT: copy mode - the current compositor failed to get its shared memory; run this script with -Fix (closes Linux GUI windows for a few seconds, keeps other WSL processes running)."
    exit 1
  }
  Write-Host "VERDICT: healthy - shared memory allocated; if a window is still blank, see references/viewer-pitfalls.md (GPU adapter / key focus)."
  exit 0
}

if ($Fix) {
  Write-Host "--- before:"
  $s = Show-State
  Write-Host "--- 1. creating /mnt/shared_memory in $Distro (persists; also the durable fix for the next boot)"
  wsl.exe -d $Distro -u root -- mkdir -p /mnt/shared_memory
  $ok = $false
  for ($i = 0; $i -lt 10; $i++) {
    $m = Get-SharedMemoryMount
    if (@($m | Where-Object { $_ -match 'virtiofs' }).Count -gt 0) { $ok = $true; break }
    Start-Sleep -Seconds 1
  }
  if (-not $ok) {
    Write-Host "    virtiofs share did not appear in the system distro; trying the tmpfs workaround in $Distro"
    wsl.exe -d $Distro -u root -- mount -t tmpfs tmpfs /mnt/shared_memory
    Start-Sleep -Seconds 3
    $m = Get-SharedMemoryMount
    if ($m.Count -eq 0) {
      Write-Host "VERDICT: fix failed - /mnt/shared_memory never showed up in the system distro. Remaining option: wsl --shutdown (kills running WSL jobs), then wsl --update if it recurs."
      exit 2
    }
  }
  Write-Host ("    system distro mount: " + ((Get-SharedMemoryMount) -join " | "))
  Write-Host "--- 2. restarting the compositor (weston) in the system distro; WSLGd respawns it"
  $old = Get-WestonPid
  if ($old) { wsl.exe --system -u root -- kill $old } else { Write-Host "    weston not running?!" }
  $new = $null
  for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    $p = Get-WestonPid
    if ($p -and $p -ne $old) { $new = $p; break }
  }
  if (-not $new) {
    Write-Host "VERDICT: fix failed - weston did not come back within 20 s. Remaining option: wsl --shutdown."
    exit 2
  }
  Start-Sleep -Seconds 3
  Write-Host "--- after:"
  $s2 = Show-State
  if ($s2.State.FailLine -gt $s2.State.StartLine) {
    Write-Host "VERDICT: fix failed - the new weston logged the shared-memory failure again. Remaining option: wsl --shutdown."
    exit 2
  }
  Write-Host "VERDICT: fixed - weston restarted (pid $new) without the shared-memory failure. Relaunch the GUI app; its title must not start with [WARN:COPY MODE]."
  exit 0
}
