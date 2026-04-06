# monitor.ps1
# Run with: .\monitor.ps1
# Or with custom path: .\monitor.ps1 -ProjectRoot "C:\Users\kashy\OneDrive\Documents\uNOS"

param(
    [string]$ProjectRoot = $PSScriptRoot,
    [int]$RefreshSeconds = 10
)

# ── Resolve all possible tuner state paths ────────────────────
$possibleRoots = @(
    $ProjectRoot,
    "C:\Users\kashy\OneDrive\Documents\uNOS",
    (Get-Location).Path
)

$tunerFiles = @(
    "finetuned\test_1h\tokenizer\tuner\tuner_state_v2.json",
    "finetuned\test_1h\basemodel\tuner\tuner_state_v2.json",
    "finetuned\test_1h\tokenizer\best_model",
    "finetuned\test_1h\basemodel\best_model",
    "finetuned\test_1h\logs\tokenizer_training_rank_0.log",
    "finetuned\test_1h\logs\basemodel_training_rank_0.log"
)

function Get-ColoredValue {
    param($value, $goodThreshold, $badThreshold, $lowerIsBetter = $true)
    if ($null -eq $value) { return "N/A" }
    $v = [float]$value
    if ($lowerIsBetter) {
        if ($v -le $goodThreshold) { 
            Write-Host "$([math]::Round($v,6))" -ForegroundColor Green -NoNewline 
        } elseif ($v -le $badThreshold) { 
            Write-Host "$([math]::Round($v,6))" -ForegroundColor Yellow -NoNewline 
        } else { 
            Write-Host "$([math]::Round($v,6))" -ForegroundColor Red -NoNewline 
        }
    }
}

function Show-TunerState {
    param([string]$JsonPath, [string]$Label)
    
    if (-not (Test-Path $JsonPath)) {
        Write-Host "  [$Label] " -ForegroundColor DarkGray -NoNewline
        Write-Host "Not started yet — file not found" -ForegroundColor DarkGray
        Write-Host "  Path checked: $JsonPath" -ForegroundColor DarkGray
        return
    }

    try {
        $raw = Get-Content $JsonPath -Raw -ErrorAction Stop
        $data = $raw | ConvertFrom-Json -ErrorAction Stop
    } catch {
        Write-Host "  [$Label] File exists but cannot read yet (training in progress)" -ForegroundColor Yellow
        return
    }

    $state = $data.current_state
    $epoch = $data.epoch
    $lossHistory = $data.loss_history
    $interventions = $data.intervention_history

    Write-Host ""
    Write-Host "  [$Label] Epoch $($epoch + 1)" -ForegroundColor Cyan
    Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkCyan

    # Loss
    if ($lossHistory -and $lossHistory.train -and $lossHistory.train.Count -gt 0) {
        $trainLosses = $lossHistory.train
        $valLosses   = $lossHistory.val
        $lastTrain   = $trainLosses[-1]
        $lastVal     = $valLosses[-1]
        $gap         = $lastVal - $lastTrain
        $gapRatio    = if ($lastTrain -gt 0) { ($lastVal - $lastTrain) / $lastTrain } else { 0 }

        Write-Host "  Loss    :" -ForegroundColor White -NoNewline
        Write-Host "  Train=" -NoNewline
        Write-Host ("{0:F6}" -f $lastTrain) -ForegroundColor Green -NoNewline
        Write-Host "  Val=" -NoNewline
        
        $valColor = if ($gapRatio -gt 0.30) { "Red" } 
                    elseif ($gapRatio -gt 0.10) { "Yellow" } 
                    else { "Green" }
        Write-Host ("{0:F6}" -f $lastVal) -ForegroundColor $valColor -NoNewline
        Write-Host "  Gap=" -NoNewline
        Write-Host ("{0:F6}" -f $gap) -ForegroundColor $valColor -NoNewline
        Write-Host ("  ({0:F3}%)" -f ($gapRatio * 100))

        # Mini loss chart (last 10 epochs)
        if ($valLosses.Count -ge 2) {
            $recent = $valLosses | Select-Object -Last 10
            Write-Host "  Val trend: " -NoNewline -ForegroundColor White
            $prev = $recent[0]
            foreach ($v in $recent) {
                $symbol = if ($v -lt $prev) { "▼" } 
                          elseif ($v -gt $prev) { "▲" } 
                          else { "─" }
                $color  = if ($v -lt $prev) { "Green" } 
                          elseif ($v -gt $prev) { "Red" } 
                          else { "Gray" }
                Write-Host $symbol -ForegroundColor $color -NoNewline
                $prev = $v
            }
            Write-Host ""
        }
    } else {
        Write-Host "  Loss    :  No loss history yet" -ForegroundColor DarkGray
    }

    # Hyperparameters
    if ($state) {
        Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkCyan
        Write-Host "  Optimizer:" -ForegroundColor White
        Write-Host ("    LR          = {0:E3}" -f [float]$state.current_lr) -ForegroundColor Yellow
        Write-Host ("    Weight Decay= {0:F4}" -f [float]$state.current_weight_decay)
        Write-Host ("    Beta1       = {0:F4}" -f [float]$state.current_beta1)
        Write-Host ("    Beta2       = {0:F4}" -f [float]$state.current_beta2)
        Write-Host ("    Eps         = {0:E2}" -f [float]$state.current_eps)
        
        Write-Host "  Regularization:" -ForegroundColor White
        Write-Host ("    Grad Clip   = {0:F3}" -f [float]$state.current_grad_clip)
        Write-Host ("    Label Smooth= {0:F4}" -f [float]$state.current_label_smoothing)
        Write-Host ("    BSQ Temp    = {0:F3}" -f [float]$state.current_bsq_inv_temperature)

        # Loss weights
        Write-Host "  Loss Weights:" -ForegroundColor White
        if ($Label -eq "TOKENIZER") {
            Write-Host ("    recon_pre={0:F3}  recon_all={1:F3}  bsq={2:F3}  recon={3:F3}" -f `
                [float]$state.current_tokenizer_recon_pre_w,
                [float]$state.current_tokenizer_recon_all_w,
                [float]$state.current_tokenizer_bsq_w,
                [float]$state.current_tokenizer_recon_w)
        } else {
            Write-Host ("    s1={0:F4}  s2={1:F4}" -f `
                [float]$state.current_s1_weight,
                [float]$state.current_s2_weight)
        }

        # Cooldowns
        $cooldowns = $state.last_intervention_epoch
        if ($cooldowns) {
            $currentEpoch = $epoch
            $onCooldown = @()
            $props = $cooldowns | Get-Member -MemberType NoteProperty
            foreach ($p in $props) {
                $lastEpoch = [int]($cooldowns.($p.Name))
                $cd = $currentEpoch - $lastEpoch
                if ($cd -lt 3) {
                    $onCooldown += "$($p.Name)($cd)"
                }
            }
            if ($onCooldown.Count -gt 0) {
                Write-Host "  Cooldowns   : $($onCooldown -join ', ')" -ForegroundColor DarkYellow
            }
        }
    }

    # Recent interventions
    if ($interventions -and $interventions.Count -gt 0) {
        Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkCyan
        Write-Host "  Recent Tuner Actions:" -ForegroundColor White
        $recent = $interventions | Select-Object -Last 5
        foreach ($iv in $recent) {
            $ep = $iv.epoch + 1
            $changes = ($iv.changes | Get-Member -MemberType NoteProperty |
                        ForEach-Object { $_.Name }) -join ", "
            Write-Host ("    Epoch {0,3}: [{1}]" -f $ep, $changes) -ForegroundColor Magenta
        }
    }
}

function Show-LogTail {
    param([string]$LogPath, [string]$Label, [int]$Lines = 5)
    
    if (-not (Test-Path $LogPath)) {
        Write-Host "  [$Label LOG] Not started" -ForegroundColor DarkGray
        return
    }
    
    Write-Host "  [$Label LOG] Last $Lines lines:" -ForegroundColor Cyan
    $tail = Get-Content $LogPath -Tail $Lines -ErrorAction SilentlyContinue
    foreach ($line in $tail) {
        # Color code by content
        if ($line -match "ERROR|error|Error") {
            Write-Host "    $line" -ForegroundColor Red
        } elseif ($line -match "TUNE|tuner|intervention") {
            Write-Host "    $line" -ForegroundColor Magenta
        } elseif ($line -match "Best model|best_val|Best val") {
            Write-Host "    $line" -ForegroundColor Green
        } elseif ($line -match "WARNING|warning") {
            Write-Host "    $line" -ForegroundColor Yellow
        } else {
            Write-Host "    $line" -ForegroundColor Gray
        }
    }
}

function Show-FileTree {
    param([string]$BasePath)
    
    Write-Host "  Project output structure:" -ForegroundColor White
    
    $checkPaths = @{
        "finetuned\test_1h"                              = "Experiment root"
        "finetuned\test_1h\tokenizer\best_model"         = "Tokenizer best model"
        "finetuned\test_1h\basemodel\best_model"         = "Basemodel best model"
        "finetuned\test_1h\tokenizer\tuner"              = "Tokenizer tuner logs"
        "finetuned\test_1h\basemodel\tuner"              = "Basemodel tuner logs"
        "finetuned\test_1h\logs"                         = "Training logs"
    }
    
    foreach ($kv in $checkPaths.GetEnumerator()) {
        $full = Join-Path $BasePath $kv.Key
        $exists = Test-Path $full
        $symbol = if ($exists) { "[OK]" } else { "[--]" }
        $color  = if ($exists) { "Green" } else { "DarkGray" }
        Write-Host ("    {0} {1}" -f $symbol, $kv.Value) -ForegroundColor $color
    }
}

# ── Find actual project root ──────────────────────────────────
$actualRoot = $null
foreach ($root in $possibleRoots) {
    if (Test-Path (Join-Path $root "train_sequential.py")) {
        $actualRoot = $root
        break
    }
}

if ($null -eq $actualRoot) {
    $actualRoot = (Get-Location).Path
    Write-Host "WARNING: Could not auto-detect project root. Using: $actualRoot" -ForegroundColor Yellow
}

Write-Host "Project root: $actualRoot" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow
Write-Host ""

# ── Main monitoring loop ──────────────────────────────────────
while ($true) {
    Clear-Host
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  NOS Training Monitor  |  $timestamp" -ForegroundColor Cyan
    Write-Host "  Refreshing every ${RefreshSeconds}s  |  Ctrl+C to stop" -ForegroundColor DarkCyan
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan

    # File tree
    Write-Host ""
    Show-FileTree -BasePath $actualRoot

    # Tokenizer tuner state
    Write-Host ""
    Write-Host "══════════════════ TOKENIZER ══════════════════" -ForegroundColor Yellow
    $tokTuner = Join-Path $actualRoot "finetuned\test_1h\tokenizer\tuner\tuner_state_v2.json"
    Show-TunerState -JsonPath $tokTuner -Label "TOKENIZER"
    
    # Tokenizer log tail
    Write-Host ""
    $tokLog = Join-Path $actualRoot "finetuned\test_1h\logs\tokenizer_training_rank_0.log"
    Show-LogTail -LogPath $tokLog -Label "TOKENIZER" -Lines 4

    # Basemodel tuner state
    Write-Host ""
    Write-Host "══════════════════ BASEMODEL ══════════════════" -ForegroundColor Yellow
    $bmTuner = Join-Path $actualRoot "finetuned\test_1h\basemodel\tuner\tuner_state_v2.json"
    Show-TunerState -JsonPath $bmTuner -Label "BASEMODEL"

    # Basemodel log tail
    Write-Host ""
    $bmLog = Join-Path $actualRoot "finetuned\test_1h\logs\basemodel_training_rank_0.log"
    Show-LogTail -LogPath $bmLog -Label "BASEMODEL" -Lines 4

    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Next refresh in ${RefreshSeconds}s..." -ForegroundColor DarkGray

    Start-Sleep -Seconds $RefreshSeconds
}