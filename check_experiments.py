#!/usr/bin/env python3
"""Check if experiments finished and fill the TSV table."""
import os, re, sys

WORK_DIR = "/home/weiying/users/mh/Hangar/TestNet/work_dirs"
TSV = os.path.join(WORK_DIR, "experiments.tsv")
TSV_UPDATED = os.path.join(WORK_DIR, "experiments_updated.tsv")

DGA10_DIR = os.path.join(WORK_DIR, "vaihingen_dga10_80_260507_163718")
DGA20_DIR = os.path.join(WORK_DIR, "vaihingen_dga20_80_260507_163730")


def is_process_running(model_type):
    """Check if training process is still running."""
    result = os.popen(f"ps aux | grep 'train.py.*{model_type}' | grep -v grep | wc -l").read().strip()
    return int(result) > 0


def extract_best(log_path):
    """Extract best epoch metrics from train.log.
    
    Validation block format:
      VALIDATION EPOCH N
      ...
      Total accuracy: XX.XXXX
      Mean F1Score: 0.XXXX
      Mean MIoU: 0.XXXX
    """
    with open(log_path) as f:
        content = f.read()
    
    # Find all validation blocks
    pattern = r"VALIDATION EPOCH (\d+).*?Total accuracy: ([\d.]+).*?Mean F1Score: ([\d.]+).*?Mean MIoU: ([\d.]+)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return None
    
    best_miou = -1
    best = {}
    
    for epoch, oa, f1, miou in matches:
        miou_f = float(miou)
        if miou_f > best_miou:
            best_miou = miou_f
            best = {
                'epoch': int(epoch),
                'miou_pct': f"{miou_f * 100:.2f}",
                'oa_pct': f"{float(oa):.2f}",
                'f1_pct': f"{float(f1) * 100:.2f}",
            }
    
    return best


def update_tsv_line(filepath, row_num, values):
    """Update a specific row in TSV file."""
    with open(filepath) as f:
        lines = f.readlines()
    
    # Build TSV line
    new_line = "\t".join(values) + "\n"
    lines[row_num - 1] = new_line  # 1-indexed rows
    
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"  Updated row {row_num}: {new_line.strip()}")


def main():
    print("=" * 60)
    print("  Experiment Check Script")
    print("=" * 60)
    
    # Check status
    dga10_running = is_process_running("dga10")
    dga20_running = is_process_running("dga20")
    
    print(f"\nDGA10 running: {dga10_running}")
    print(f"DGA20 running: {dga20_running}")
    
    if dga10_running or dga20_running:
        print("\n⚠️  Experiments still running, skipping table update.")
        
        # Show partial results if log exists
        for name, exp_dir in [("DGA10", DGA10_DIR), ("DGA20", DGA20_DIR)]:
            log_path = os.path.join(exp_dir, "train.log")
            if os.path.exists(log_path):
                best = extract_best(log_path)
                if best:
                    print(f"\n  {name} partial best: epoch={best['epoch']}, "
                          f"mIoU={best['miou_pct']}%, OA={best['oa_pct']}%, F1={best['f1_pct']}%")
        return
    
    print("\n✅  Both experiments finished!")
    
    results = {}
    for name, exp_dir in [("DGA10", DGA10_DIR), ("DGA20", DGA20_DIR)]:
        log_path = os.path.join(exp_dir, "train.log")
        if not os.path.exists(log_path):
            print(f"❌  {name}: train.log not found!")
            continue
        
        best = extract_best(log_path)
        if not best:
            print(f"❌  {name}: Could not parse metrics from log!")
            continue
        
        results[name] = best
        print(f"\n  {name} best: epoch={best['epoch']}")
        print(f"    mIoU: {best['miou_pct']}%")
        print(f"    OA:   {best['oa_pct']}%")
        print(f"    F1:   {best['f1_pct']}%")
    
    if "DGA10" not in results and "DGA20" not in results:
        print("No results to write. Aborting.")
        return
    
    print("\n" + "-" * 60)
    print("Updating experiments.tsv...")
    
    # experiments.tsv columns: Date\tDataset\tModules\tLoss\tSeed\tmIoU\tOA\tF1\tBestE\tNote
    if "DGA10" in results:
        r = results["DGA10"]
        update_tsv_line(TSV, 12, [
            "2026-05-08", "Vaihingen", "DGA10", "CE", "80",
            r["miou_pct"], r["oa_pct"], r["f1_pct"], str(r["epoch"]), ""
        ])
    if "DGA20" in results:
        r = results["DGA20"]
        update_tsv_line(TSV, 13, [
            "2026-05-08", "Vaihingen", "DGA20", "CE", "80",
            r["miou_pct"], r["oa_pct"], r["f1_pct"], str(r["epoch"]), ""
        ])
    
    print("\nUpdating experiments_updated.tsv...")
    # experiments_updated.tsv columns: Date\tDataset\tModules\tLoss\tSeed\tmIoU\tOA\tF1\tBestE\tNote\tCommand
    if "DGA10" in results:
        r = results["DGA10"]
        update_tsv_line(TSV_UPDATED, 12, [
            "2026-05-08", "Vaihingen", "DGA10", "CE", "80",
            r["miou_pct"], r["oa_pct"], r["f1_pct"], str(r["epoch"]), "",
            "python train.py --config configs/train_config.jsonc --model-type mfnet_unetformer_dga10"
        ])
    if "DGA20" in results:
        r = results["DGA20"]
        update_tsv_line(TSV_UPDATED, 13, [
            "2026-05-08", "Vaihingen", "DGA20", "CE", "80",
            r["miou_pct"], r["oa_pct"], r["f1_pct"], str(r["epoch"]), "",
            "python train.py --config configs/train_config.jsonc --model-type mfnet_unetformer_dga20"
        ])
    
    print("\n" + "=" * 60)
    print("✅  Done! TSV files updated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
