import os
import shutil
from pathlib import Path

cache_path = Path("D:/huggingface_cache/hub/models--genmo--mochi-1-preview")

print("🔍 Analyzing cache...")

# Find and remove incomplete downloads
incomplete_files = list(cache_path.rglob("*.incomplete"))
print(f"\n❌ Found {len(incomplete_files)} incomplete download files")
total_incomplete = 0
for f in incomplete_files:
    size = f.stat().st_size / (1024**3)
    total_incomplete += size
    print(f"   Deleting: {f.name} ({size:.2f} GB)")
    f.unlink()

print(f"💾 Freed {total_incomplete:.2f} GB from incomplete downloads")

# Find and remove lock files
lock_files = list(cache_path.rglob("*.lock"))
print(f"\n🔒 Found {len(lock_files)} lock files")
for f in lock_files:
    print(f"   Deleting: {f.name}")
    f.unlink()

# Check for duplicate snapshots (keep only latest)
snapshots_dir = cache_path / "snapshots"
if snapshots_dir.exists():
    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    print(f"\n📸 Found {len(snapshots)} snapshot(s)")
    
    if len(snapshots) > 1:
        # Sort by modification time, keep newest
        snapshots_sorted = sorted(snapshots, key=lambda x: x.stat().st_mtime, reverse=True)
        latest = snapshots_sorted[0]
        print(f"   ✅ Keeping latest: {latest.name}")
        
        total_removed = 0
        for old_snapshot in snapshots_sorted[1:]:
            size = sum(f.stat().st_size for f in old_snapshot.rglob('*') if f.is_file()) / (1024**3)
            total_removed += size
            print(f"   ❌ Deleting old: {old_snapshot.name} ({size:.2f} GB)")
            shutil.rmtree(old_snapshot)
        
        print(f"💾 Freed {total_removed:.2f} GB from old snapshots")

# Calculate final size
total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / (1024**3)
file_count = len([f for f in cache_path.rglob('*') if f.is_file()])

print(f"\n✅ Cleanup complete!")
print(f"📊 Final cache size: {total_size:.2f} GB")
print(f"📁 Total files: {file_count}")
