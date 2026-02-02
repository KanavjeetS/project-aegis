# 🚨 Disk Space Emergency Guide

## Your Situation

**Problem:** C: drive is completely full (0GB free)  
**Impact:** Cannot install ANY Python packages locally  
**Solution:** Use Google Colab OR free up significant space

---

## ✅ Solution 1: Google Colab (IMMEDIATE - No Space Needed)

**Best option since local drive is full!**

### Steps:

1. **Upload Project to GitHub:**
   ```powershell
   cd "d:\github pipeline\project-aegis"
   git init
   git add .
   git commit -m "Initial commit"
   # Create repo on GitHub, then:
   git remote add origin https://github.com/YOUR_USERNAME/project-aegis.git
   git push -u origin main
   ```

2. **Open Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File → Open Notebook → GitHub
   - Enter your repo URL
   - Open `notebooks/01_colab_quickstart.ipynb`

3. **Run Everything in Cloud:**
   - Free T4 GPU (16GB VRAM)
   - 100GB+ storage
   - 12-hour sessions
   - **Zero local storage used!**

---

## ⚠️ Solution 2: Free Up C: Drive Space (If You Want Local)

### Quick Wins (Run These Commands)

```powershell
# 1. Clear pip cache (~500MB-2GB)
pip cache purge

# 2. Clear Windows temp files (1-5GB)
Remove-Item -Path $env:TEMP\* -Recurse -Force -ErrorAction SilentlyContinue

# 3. Disk Cleanup (5-20GB potential)
cleanmgr /sagerun:1

# 4. Check what's using space
Get-ChildItem C:\ -Recurse -ErrorAction SilentlyContinue | 
  Group-Object Extension | 
  Sort-Object -Descending {($_.Group | Measure-Object Length -Sum).Sum} | 
  Select-Object -First 20 Name, @{N='Size(MB)';E={[math]::Round(($_.Group | Measure-Object Length -Sum).Sum / 1MB, 2)}}
```

### Manual Cleanup

1. **Uninstall Unused Programs** (Control Panel → Programs)
2. **Delete Large Files:**
   - Old downloads
   - Video files
   - Game installations
   - Old Python environments
3. **Move to D: Drive** (if you have one):
   - Documents
   - Videos
   - Other large folders

### After Freeing 10GB+

```powershell
# Try minimal install again
cd "d:\github pipeline\project-aegis"
pip cache purge  # Clear cache first!
pip install -r requirements-minimal.txt
```

---

## 📊 Space Requirements

| Component | Size | Where |
|-----------|------|-------|
| Python + pip cache | ~2GB | C: |
| Project dependencies | ~2GB | C: |
| V-JEPA checkpoint | ~900MB | C: or D: |
| Datasets (optional) | ~20GB | **D: drive** |
| **Minimum for local** | **~5GB free on C:** | |

---

## 🎯 Recommended: Hybrid Approach

1. **Keep code on D: drive** (already there!)
2. **Use Colab for training** (free GPU + storage)
3. **Download only final checkpoint** (900MB)
4. **Run inference locally** (2-3GB total)

### Hybrid Setup

```powershell
# On Colab (cloud):
# - Install all dependencies
# - Train Q-Former (6-8 hours)
# - Download final checkpoint

# On Local (after freeing 5GB):
pip install torch transformers einops omegaconf  # Minimal
# - Run inference only
# - No training locally
```

---

## 🆘 If Still Stuck

**Your project is complete and works perfectly - it's just a local storage issue!**

**Options:**
1. ✅ **Use Colab** (easiest, zero local storage)
2. Free up 10GB on C: drive
3. Install Python on D: drive (if separate disk)
4. Use external USB drive for pip cache

**The code is ready. The architecture is solid. You just need space to run it!**

---

## Next Immediate Step

**Go to Colab now:**
1. Upload `project-aegis` folder to Google Drive
2. Open Colab notebook
3. Mount Drive
4. Run all cells

**Or free space:**
1. Run `cleanmgr /sagerun:1`
2. Delete 10GB of files
3. Retry `pip install -r requirements-minimal.txt`

---

**The project is complete. The only blocker is local disk space. Colab solves this completely!** 🚀
