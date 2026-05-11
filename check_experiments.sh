#!/bin/bash
# Script to check if experiments finished and fill the TSV table
WORK_DIR="/home/weiying/users/mh/Hangar/TestNet/work_dirs"
TSV="$WORK_DIR/experiments.tsv"
TSV_UPDATED="$WORK_DIR/experiments_updated.tsv"

DGA10_DIR="$WORK_DIR/vaihingen_dga10_80_260507_163718"
DGA20_DIR="$WORK_DIR/vaihingen_dga20_80_260507_163730"

# Check if processes are still running
DGA10_RUNNING=$(ps aux | grep "train.py.*dga10" | grep -v grep | wc -l)
DGA20_RUNNING=$(ps aux | grep "train.py.*dga20" | grep -v grep | wc -l)

echo "=== Status ==="
echo "DGA10 PID found: $DGA10_RUNNING"
echo "DGA20 PID found: $DGA20_RUNNING"

if [ "$DGA10_RUNNING" -gt 0 ] || [ "$DGA20_RUNNING" -gt 0 ]; then
    echo "Experiments still running, skipping table update."
    exit 0
fi

echo "=== Both finished! Extracting metrics ==="

# --- Extract DGA10 metrics ---
extract_best() {
    local LOG="$1/train.log"
    local NAME="$2"
    
    # Parse best epoch and metrics
    # Find the validation block where MIoU_best was last updated
    local BEST_LINE=$(grep -n "MIoU_best" "$LOG" | tail -1)
    local BEST_EPOCH=$(echo "$BEST_LINE" | grep -oP 'MIoU_best: \K[\d.]+')
    
    # Actually, let me find which epoch produced the best
    # The MIoU_best line appears AFTER the validation block
    # Let's go backwards to find the corresponding validation epoch
    local BEST_LINE_NUM=$(echo "$BEST_LINE" | cut -d: -f1)
    local BEST_EPOCH_NUM=$(sed -n "1,${BEST_LINE_NUM}p" "$LOG" | grep "VALIDATION EPOCH" | tail -1 | grep -oP '\d+')
    
    # Get the validation metrics for that epoch
    local VAL_BLOCK=$(grep -B2 "VALIDATION EPOCH $BEST_EPOCH_NUM" "$LOG" | head -1 | grep -oP '\d+')
    
    # Better: get the full validation report
    local OA=$(sed -n "/VALIDATION EPOCH $BEST_EPOCH_NUM/,/\[Confusion/p" "$LOG" | grep "Total accuracy" | awk '{print $3}')
    local F1=$(sed -n "/VALIDATION EPOCH $BEST_EPOCH_NUM/,/\[Confusion/p" "$LOG" | grep "Mean F1Score" | awk '{print $3}')
    local MIOU=$(sed -n "/VALIDATION EPOCH $BEST_EPOCH_NUM/,/\[Confusion/p" "$LOG" | grep "Mean MIoU" | awk '{print $3}')
    
    # Convert to percentage (2 decimal places)
    local MIOU_PCT=$(printf "%.2f" $(echo "$MIOU * 100" | bc -l))
    local OA_PCT=$(printf "%.2f" $OA)
    local F1_PCT=$(printf "%.2f" $(echo "$F1 * 100" | bc -l))
    
    echo "$NAME|$MIOU_PCT|$OA_PCT|$F1_PCT|$BEST_EPOCH_NUM"
}

RESULT_DGA10=$(extract_best "$DGA10_DIR" "DGA10")
RESULT_DGA20=$(extract_best "$DGA20_DIR" "DGA20")

echo ""
echo "=== Extracted Results ==="
echo "$RESULT_DGA10"
echo "$RESULT_DGA20"

# Parse results
DGA10_MIOU=$(echo "$RESULT_DGA10" | cut -d'|' -f2)
DGA10_OA=$(echo "$RESULT_DGA10" | cut -d'|' -f3)
DGA10_F1=$(echo "$RESULT_DGA10" | cut -d'|' -f4)
DGA10_BESTE=$(echo "$RESULT_DGA10" | cut -d'|' -f5)

DGA20_MIOU=$(echo "$RESULT_DGA20" | cut -d'|' -f2)
DGA20_OA=$(echo "$RESULT_DGA20" | cut -d'|' -f3)
DGA20_F1=$(echo "$RESULT_DGA20" | cut -d'|' -f4)
DGA20_BESTE=$(echo "$RESULT_DGA20" | cut -d'|' -f5)

echo ""
echo "=== Updating TSV files ==="

# Update experiments.tsv (rows 12 and 13)
# Row 12: DGA10
sed -i "12s/.*/2026-05-07\tVaihingen\tDGA10\tCE\t80\t$DGA10_MIOU\t$DGA10_OA\t$DGA10_F1\t$DGA10_BESTE\t/" "$TSV"
# Row 13: DGA20
sed -i "13s/.*/2026-05-07\tVaihingen\tDGA20\tCE\t80\t$DGA20_MIOU\t$DGA20_OA\t$DGA20_F1\t$DGA20_BESTE\t/" "$TSV"

# Update experiments_updated.tsv (rows 12 and 13)
# Note: these rows have different columns (includes Command column)
# Row 12: DGA10  
sed -i "12s/.*/2026-05-08\tVaihingen\tDGA10\tCE\t80\t$DGA10_MIOU\t$DGA10_OA\t$DGA10_F1\t$DGA10_BESTE\t\tpython train.py --config configs\/train_config.jsonc --model-type mfnet_unetformer_dga10/" "$TSV_UPDATED"
# Row 13: DGA20
sed -i "13s/.*/2026-05-08\tVaihingen\tDGA20\tCE\t80\t$DGA20_MIOU\t$DGA20_OA\t$DGA20_F1\t$DGA20_BESTE\t\tpython train.py --config configs\/train_config.jsonc --model-type mfnet_unetformer_dga20/" "$TSV_UPDATED"

echo "=== Done! ==="
echo ""
echo "experiments.tsv row 12: $(sed -n '12p' $TSV)"
echo "experiments.tsv row 13: $(sed -n '13p' $TSV)"
echo "experiments_updated.tsv row 12: $(sed -n '12p' $TSV_UPDATED)"
echo "experiments_updated.tsv row 13: $(sed -n '13p' $TSV_UPDATED)"
