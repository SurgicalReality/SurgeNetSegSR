#!/bin/bash
# Video Preprocessing Pipeline Script
# This script orchestrates the complete video preprocessing workflow:
# 1. Crop video to remove borders and bars
# 2. Split video into clips
# 3. Convert to DAVIS dataset format

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
CLIP_DURATION=30
DAVIS_OUTPUT_PATH="./custom/data/DAVIS"
LEFT=0
RIGHT=0
TOP=0
BOTTOM=0
SKIP_CROP=false
SKIP_SPLIT=false
SKIP_DAVIS=false

# Usage function
usage() {
    echo "Usage: $0 --video <path> [options]"
    echo ""
    echo "Required:"
    echo "  --video <path>           Path to input video file"
    echo ""
    echo "Optional:"
    echo "  --crop-config <path>     Path to crop configuration JSON file"
    echo "  --left <pixels>          Pixels to remove from left (default: 0)"
    echo "  --right <pixels>         Pixels to remove from right (default: 0)"
    echo "  --top <pixels>           Pixels to remove from top (default: 0)"
    echo "  --bottom <pixels>        Pixels to remove from bottom (default: 0)"
    echo "  --clip-duration <sec>    Clip duration in seconds (default: 30)"
    echo "  --output-dir <path>      Output directory for clips"
    echo "  --workspace <path>       Workspace path for DAVIS conversion"
    echo "  --davis-output <path>    DAVIS output path (default: ./custom/data/DAVIS)"
    echo "  --skip-crop              Skip cropping step"
    echo "  --skip-split             Skip splitting step"
    echo "  --skip-davis             Skip DAVIS conversion step"
    echo "  --help                   Show this help message"
    exit 1
}

# Print colored messages
print_step() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --crop-config)
            CROP_CONFIG="$2"
            shift 2
            ;;
        --left)
            LEFT="$2"
            shift 2
            ;;
        --right)
            RIGHT="$2"
            shift 2
            ;;
        --top)
            TOP="$2"
            shift 2
            ;;
        --bottom)
            BOTTOM="$2"
            shift 2
            ;;
        --clip-duration)
            CLIP_DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE_PATH="$2"
            shift 2
            ;;
        --davis-output)
            DAVIS_OUTPUT_PATH="$2"
            shift 2
            ;;
        --skip-crop)
            SKIP_CROP=true
            shift
            ;;
        --skip-split)
            SKIP_SPLIT=true
            shift
            ;;
        --skip-davis)
            SKIP_DAVIS=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$VIDEO_PATH" ]; then
    print_error "Video path is required"
    usage
fi

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    print_error "Video file not found: $VIDEO_PATH"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')

echo -e "${YELLOW}Video Preprocessing Pipeline${NC}"
echo -e "${YELLOW}Input Video: $VIDEO_PATH${NC}"
echo ""

# Initialize paths
CROPPED_VIDEO_PATH="$VIDEO_PATH"

# ============================================================
# STEP 1: CROP VIDEO
# ============================================================
if [ "$SKIP_CROP" = false ]; then
    print_step "STEP 1: Cropping Video"
    
    CROP_ARGS="crop --video \"$VIDEO_PATH\""
    
    # Use config file or manual parameters
    if [ -n "$CROP_CONFIG" ] && [ -f "$CROP_CONFIG" ]; then
        echo "Using crop configuration from: $CROP_CONFIG"
        CROP_ARGS="$CROP_ARGS --config \"$CROP_CONFIG\""
    elif [ $LEFT -gt 0 ] || [ $RIGHT -gt 0 ] || [ $TOP -gt 0 ] || [ $BOTTOM -gt 0 ]; then
        echo "Using manual crop parameters: L=$LEFT, R=$RIGHT, T=$TOP, B=$BOTTOM"
        CROP_ARGS="$CROP_ARGS --left $LEFT --right $RIGHT --top $TOP --bottom $BOTTOM"
    else
        echo "No crop parameters provided, skipping crop step."
        echo "To crop, provide either --crop-config or manual crop values"
        SKIP_CROP=true
    fi
    
    # Only run if we have crop parameters
    if [ "$SKIP_CROP" = false ]; then
        CROPPED_VIDEO_PATH="$(dirname "$VIDEO_PATH")/${VIDEO_NAME}_cropped.mp4"
        CROP_ARGS="$CROP_ARGS --output \"$CROPPED_VIDEO_PATH\""
        
        CROP_CMD="python \"$SCRIPT_DIR/preprocess_videos.py\" $CROP_ARGS"
        echo "Executing: $CROP_CMD"
        eval $CROP_CMD
        
        if [ $? -ne 0 ]; then
            print_error "Cropping failed"
            exit 1
        fi
        
        print_success "Video cropped successfully: $CROPPED_VIDEO_PATH"
    fi
else
    echo -e "${YELLOW}Skipping crop step (--skip-crop specified)${NC}"
fi

# ============================================================
# STEP 2: SPLIT VIDEO INTO CLIPS
# ============================================================
if [ "$SKIP_SPLIT" = false ]; then
    print_step "STEP 2: Splitting Video into Clips"
    
    # Determine output directory for clips
    if [ -z "$OUTPUT_DIR" ]; then
        CLIPS_DIR="$(dirname "$CROPPED_VIDEO_PATH")/clips"
    else
        CLIPS_DIR="$OUTPUT_DIR"
    fi
    
    SPLIT_CMD="python \"$SCRIPT_DIR/split_video.py\" \"$CROPPED_VIDEO_PATH\" --clip_duration $CLIP_DURATION --output_dir \"$CLIPS_DIR\""
    echo "Executing: $SPLIT_CMD"
    eval $SPLIT_CMD
    
    if [ $? -ne 0 ]; then
        print_error "Video splitting failed"
        exit 1
    fi
    
    print_success "Video split into clips: $CLIPS_DIR"
else
    echo -e "${YELLOW}Skipping split step (--skip-split specified)${NC}"
    if [ -n "$OUTPUT_DIR" ]; then
        CLIPS_DIR="$OUTPUT_DIR"
    fi
fi

# ============================================================
# STEP 3: CONVERT TO DAVIS DATASET FORMAT
# ============================================================
if [ "$SKIP_DAVIS" = false ]; then
    print_step "STEP 3: Converting to DAVIS Dataset Format"
    
    if [ -z "$CLIPS_DIR" ]; then
        echo -e "${YELLOW}Warning: No clips directory available. Skipping DAVIS conversion.${NC}"
        echo -e "${YELLOW}You can run the conversion manually later using convert_to_DAVIS_dataset.py${NC}"
    elif [ -z "$WORKSPACE_PATH" ]; then
        echo -e "${YELLOW}Warning: No workspace path provided (--workspace). Skipping DAVIS conversion.${NC}"
        echo -e "${YELLOW}After processing videos in GUI, run:${NC}"
        echo "  python custom/convert_to_DAVIS_dataset.py --workspace_path <workspace> --video_names <names> --out_path $DAVIS_OUTPUT_PATH"
    else
        # Get all clip names (without extension)
        VIDEO_NAMES=$(find "$CLIPS_DIR" -name "*.mp4" -exec basename {} .mp4 \; | paste -sd "," -)
        
        if [ -z "$VIDEO_NAMES" ]; then
            print_error "No video clips found in $CLIPS_DIR"
        else
            DAVIS_CMD="python \"$SCRIPT_DIR/convert_to_DAVIS_dataset.py\" --workspace_path \"$WORKSPACE_PATH\" --video_names \"$VIDEO_NAMES\" --out_path \"$DAVIS_OUTPUT_PATH\""
            echo "Executing: $DAVIS_CMD"
            eval $DAVIS_CMD
            
            if [ $? -ne 0 ]; then
                print_error "DAVIS conversion failed"
                exit 1
            fi
            
            print_success "Converted to DAVIS format: $DAVIS_OUTPUT_PATH"
        fi
    fi
else
    echo -e "${YELLOW}Skipping DAVIS conversion step (--skip-davis specified)${NC}"
fi

# ============================================================
# PIPELINE COMPLETE
# ============================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Summary
echo -e "${CYAN}Summary:${NC}"
if [ "$SKIP_CROP" = false ] && [ "$CROPPED_VIDEO_PATH" != "$VIDEO_PATH" ]; then
    echo "  Cropped Video: $CROPPED_VIDEO_PATH"
fi
if [ "$SKIP_SPLIT" = false ] && [ -n "$CLIPS_DIR" ]; then
    echo "  Video Clips: $CLIPS_DIR"
fi
if [ "$SKIP_DAVIS" = false ] && [ -n "$WORKSPACE_PATH" ]; then
    echo "  DAVIS Dataset: $DAVIS_OUTPUT_PATH"
fi

echo ""
echo -e "${CYAN}Next Steps:${NC}"
if [ "$SKIP_DAVIS" = true ] || [ -z "$WORKSPACE_PATH" ]; then
    echo "  1. Process video clips in the GUI workspace"
    echo "  2. Run DAVIS conversion:"
    echo "     python custom/convert_to_DAVIS_dataset.py --workspace_path <workspace> --video_names <names> --out_path $DAVIS_OUTPUT_PATH"
fi

exit 0
