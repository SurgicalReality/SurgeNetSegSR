# Video Preprocessing Pipeline Script
# This script orchestrates the complete video preprocessing workflow:
# 1. Crop video to remove borders and bars
# 2. Split video into clips
# 3. Convert to DAVIS dataset format

param(
    [Parameter(Mandatory=$true, HelpMessage="Path to input video file")]
    [string]$VideoPath,
    
    [Parameter(Mandatory=$false, HelpMessage="Path to crop configuration JSON file")]
    [string]$CropConfig = "",
    
    [Parameter(Mandatory=$false, HelpMessage="Clip duration in seconds")]
    [int]$ClipDuration = 30,
    
    [Parameter(Mandatory=$false, HelpMessage="Output directory for clips")]
    [string]$OutputDir = "",
    
    [Parameter(Mandatory=$false, HelpMessage="Workspace path for DAVIS conversion")]
    [string]$WorkspacePath = "",
    
    [Parameter(Mandatory=$false, HelpMessage="DAVIS output path")]
    [string]$DavisOutputPath = ".\custom\data\DAVIS",
    
    [Parameter(Mandatory=$false, HelpMessage="Manual crop values - Left pixels")]
    [int]$Left = 0,
    
    [Parameter(Mandatory=$false, HelpMessage="Manual crop values - Right pixels")]
    [int]$Right = 0,
    
    [Parameter(Mandatory=$false, HelpMessage="Manual crop values - Top pixels")]
    [int]$Top = 0,
    
    [Parameter(Mandatory=$false, HelpMessage="Manual crop values - Bottom pixels")]
    [int]$Bottom = 0,
    
    [Parameter(Mandatory=$false, HelpMessage="Skip cropping step")]
    [switch]$SkipCrop,
    
    [Parameter(Mandatory=$false, HelpMessage="Skip splitting step")]
    [switch]$SkipSplit,
    
    [Parameter(Mandatory=$false, HelpMessage="Skip DAVIS conversion step")]
    [switch]$SkipDavisConversion
)

# Color output functions
function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Check if video file exists
if (-not (Test-Path $VideoPath)) {
    Write-Error-Custom "Video file not found: $VideoPath"
    exit 1
}

$VideoPath = Resolve-Path $VideoPath
$VideoName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Video Preprocessing Pipeline" -ForegroundColor Yellow
Write-Host "Input Video: $VideoPath" -ForegroundColor Yellow
Write-Host ""

# Initialize paths
$CroppedVideoPath = $VideoPath
$ClipsDir = ""

# ============================================================
# STEP 1: CROP VIDEO
# ============================================================
if (-not $SkipCrop) {
    Write-Step "STEP 1: Cropping Video"
    
    $cropArgs = @(
        "crop",
        "--video", "`"$VideoPath`""
    )
    
    # Use config file or manual parameters
    if ($CropConfig -ne "" -and (Test-Path $CropConfig)) {
        Write-Host "Using crop configuration from: $CropConfig"
        $cropArgs += "--config", "`"$CropConfig`""
    }
    elseif ($Left -gt 0 -or $Right -gt 0 -or $Top -gt 0 -or $Bottom -gt 0) {
        Write-Host "Using manual crop parameters: L=$Left, R=$Right, T=$Top, B=$Bottom"
        $cropArgs += "--left", $Left
        $cropArgs += "--right", $Right
        $cropArgs += "--top", $Top
        $cropArgs += "--bottom", $Bottom
    }
    else {
        Write-Host "No crop parameters provided, skipping crop step."
        Write-Host "To crop, provide either --CropConfig or manual crop values (--Left, --Right, --Top, --Bottom)"
    }
    
    # Only run if we have crop parameters
    if ($cropArgs.Count -gt 3) {
        $CroppedVideoPath = Join-Path (Split-Path $VideoPath) "${VideoName}_cropped.mp4"
        $cropArgs += "--output", "`"$CroppedVideoPath`""
        
        $cropCommand = "python `"$ScriptDir\preprocess_videos.py`" " + ($cropArgs -join " ")
        Write-Host "Executing: $cropCommand"
        Invoke-Expression $cropCommand
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Cropping failed with exit code $LASTEXITCODE"
            exit 1
        }
        
        Write-Success "Video cropped successfully: $CroppedVideoPath"
    }
} else {
    Write-Host "Skipping crop step (--SkipCrop specified)" -ForegroundColor Yellow
}

# ============================================================
# STEP 2: SPLIT VIDEO INTO CLIPS
# ============================================================
if (-not $SkipSplit) {
    Write-Step "STEP 2: Splitting Video into Clips"
    
    # Determine output directory for clips
    if ($OutputDir -eq "") {
        $ClipsDir = Join-Path (Split-Path $CroppedVideoPath) "clips"
    } else {
        $ClipsDir = $OutputDir
    }
    
    $splitCommand = "python `"$ScriptDir\split_video.py`" `"$CroppedVideoPath`" --clip_duration $ClipDuration --output_dir `"$ClipsDir`""
    Write-Host "Executing: $splitCommand"
    Invoke-Expression $splitCommand
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Video splitting failed with exit code $LASTEXITCODE"
        exit 1
    }
    
    Write-Success "Video split into clips: $ClipsDir"
} else {
    Write-Host "Skipping split step (--SkipSplit specified)" -ForegroundColor Yellow
    if ($OutputDir -ne "") {
        $ClipsDir = $OutputDir
    }
}

# ============================================================
# STEP 3: CONVERT TO DAVIS DATASET FORMAT
# ============================================================
if (-not $SkipDavisConversion) {
    Write-Step "STEP 3: Converting to DAVIS Dataset Format"
    
    if ($ClipsDir -eq "") {
        Write-Host "Warning: No clips directory available. Skipping DAVIS conversion." -ForegroundColor Yellow
        Write-Host "You can run the conversion manually later using convert_to_DAVIS_dataset.py" -ForegroundColor Yellow
    }
    elseif ($WorkspacePath -eq "") {
        Write-Host "Warning: No workspace path provided (--WorkspacePath). Skipping DAVIS conversion." -ForegroundColor Yellow
        Write-Host "After processing videos in GUI, run:" -ForegroundColor Yellow
        Write-Host "  python custom\convert_to_DAVIS_dataset.py --workspace_path <workspace> --video_names <names> --out_path $DavisOutputPath" -ForegroundColor Yellow
    }
    else {
        # Get all clip names (without extension)
        $clipFiles = Get-ChildItem -Path $ClipsDir -Filter "*.mp4"
        $videoNames = ($clipFiles | ForEach-Object { $_.BaseName }) -join ","
        
        if ($videoNames -eq "") {
            Write-Error-Custom "No video clips found in $ClipsDir"
        } else {
            $davisCommand = "python `"$ScriptDir\convert_to_DAVIS_dataset.py`" --workspace_path `"$WorkspacePath`" --video_names `"$videoNames`" --out_path `"$DavisOutputPath`""
            Write-Host "Executing: $davisCommand"
            Invoke-Expression $davisCommand
            
            if ($LASTEXITCODE -ne 0) {
                Write-Error-Custom "DAVIS conversion failed with exit code $LASTEXITCODE"
                exit 1
            }
            
            Write-Success "Converted to DAVIS format: $DavisOutputPath"
        }
    }
} else {
    Write-Host "Skipping DAVIS conversion step (--SkipDavisConversion specified)" -ForegroundColor Yellow
}

# ============================================================
# PIPELINE COMPLETE
# ============================================================
Write-Host "`n" 
Write-Host "========================================" -ForegroundColor Green
Write-Host "   PIPELINE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "Summary:" -ForegroundColor Cyan
if (-not $SkipCrop -and $CroppedVideoPath -ne $VideoPath) {
    Write-Host "  Cropped Video: $CroppedVideoPath" -ForegroundColor White
}
if (-not $SkipSplit -and $ClipsDir -ne "") {
    Write-Host "  Video Clips: $ClipsDir" -ForegroundColor White
}
if (-not $SkipDavisConversion -and $WorkspacePath -ne "") {
    Write-Host "  DAVIS Dataset: $DavisOutputPath" -ForegroundColor White
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
if ($SkipDavisConversion -or $WorkspacePath -eq "") {
    Write-Host "  1. Process video clips in the GUI workspace" -ForegroundColor White
    Write-Host "  2. Run DAVIS conversion:" -ForegroundColor White
    Write-Host "     python custom\convert_to_DAVIS_dataset.py --workspace_path <workspace> --video_names <names> --out_path $DavisOutputPath" -ForegroundColor Gray
}

exit 0
