"""
Project Cleanup Script for HMM Forex Regime Detection

This script removes temporary files created during development/debugging
and organizes the project into a clean, production-ready state.
"""

import os
from pathlib import Path
import shutil

def cleanup():
    print("=" * 70)
    print("🧹 HMM PROJECT CLEANUP")
    print("=" * 70)
    print()
    
    # Define files/directories to remove
    temporary_files = [
        'add_labels_to_existing_models.py',
        'fix_label_mismatch.py',
        'STATE_LABELS_UPDATE.md',
        'LABELS_UPDATE_COMPLETE.md',
        'LABEL_FIX_SUMMARY.md',
    ]
    
    # Temporary documentation files (development artifacts)
    temp_docs = [
        'IMPLEMENTATION_COMPLETE.md',
        'IMPLEMENTATION_SUMMARY.md',
        'MIGRATION_GUIDE.md',
        'PRIORITY1_COMPLETE.md',
        'QUICKSTART_ENHANCED.md',
    ]
    
    # Test files that might not be needed
    optional_test_files = [
        'test_basic.py',
        'test_ingestion.py',
        'run_tests.py',
        'validate_system.py',
    ]
    
    # Optional result/log files
    optional_results = [
        'backtest_EUR_USD_adaptive.json',
        'backtest_EUR_USD_momentum.json',
        'backtest_EUR_USD_simple.json',
        'predictions_EUR_USD.json',  # Duplicate of predictions.json
        'workflow_summary.json',
        'hmm_system.log',
    ]
    
    deleted_count = 0
    
    # Remove temporary fix files
    print("📁 REMOVING TEMPORARY FIX FILES:")
    print("-" * 70)
    for file in temporary_files:
        if Path(file).exists():
            os.remove(file)
            print(f"✅ Deleted: {file}")
            deleted_count += 1
        else:
            print(f"⏭️  Not found: {file}")
    print()
    
    # Remove temporary documentation
    print("📄 REMOVING TEMPORARY DOCUMENTATION:")
    print("-" * 70)
    for file in temp_docs:
        if Path(file).exists():
            os.remove(file)
            print(f"✅ Deleted: {file}")
            deleted_count += 1
        else:
            print(f"⏭️  Not found: {file}")
    print()
    
    # Remove optional test files
    print("🧪 REMOVING OPTIONAL TEST FILES:")
    print("-" * 70)
    for file in optional_test_files:
        if Path(file).exists():
            os.remove(file)
            print(f"✅ Deleted: {file}")
            deleted_count += 1
        else:
            print(f"⏭️  Not found: {file}")
    print()
    
    # Remove optional result files
    print("📊 REMOVING OPTIONAL RESULT FILES:")
    print("-" * 70)
    for file in optional_results:
        if Path(file).exists():
            os.remove(file)
            print(f"✅ Deleted: {file}")
            deleted_count += 1
        else:
            print(f"⏭️  Not found: {file}")
    print()
    
    # Clean up empty directories
    print("📂 CLEANING UP DIRECTORIES:")
    print("-" * 70)
    dirs_to_check = ['logs', 'results', 'config']
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            contents = list(dir_path.iterdir())
            if not contents or all(f.name in ['.gitkeep', '__pycache__'] for f in contents):
                shutil.rmtree(dir_path)
                print(f"✅ Removed empty directory: {dir_name}/")
                deleted_count += 1
            else:
                print(f"⏭️  Kept non-empty directory: {dir_name}/ ({len(contents)} items)")
        else:
            print(f"⏭️  Directory not found: {dir_name}/")
    print()
    
    # Clean up cache directories
    print("🗑️  REMOVING CACHE FILES:")
    print("-" * 70)
    cache_dirs = ['.pytest_cache', '__pycache__']
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
            print(f"✅ Deleted: {cache_dir}/")
            deleted_count += 1
    
    # Remove pycache in subdirectories
    for pycache in Path('.').rglob('__pycache__'):
        shutil.rmtree(pycache)
        print(f"✅ Deleted: {pycache}")
        deleted_count += 1
    print()
    
    # Summary
    print("=" * 70)
    print(f"✅ CLEANUP COMPLETE! Removed {deleted_count} items")
    print("=" * 70)
    print()
    
    # Show essential files that remain
    print("📋 ESSENTIAL FILES KEPT:")
    print("-" * 70)
    essential_files = [
        'README.md',
        'requirements.txt',
        'doit.py',
        'hmm_cli.py',
        'setup.py',
        'pytest.ini',
        '.gitignore',
        'predictions.json',
        'HMM_Model_Evaluation.ipynb',
    ]
    
    for file in essential_files:
        status = "✅" if Path(file).exists() else "❌ MISSING"
        print(f"{status} {file}")
    print()
    
    print("📂 ESSENTIAL DIRECTORIES KEPT:")
    print("-" * 70)
    essential_dirs = ['src', 'data', 'models', 'notebooks', 'tests']
    for dir_name in essential_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            file_count = len(list(dir_path.rglob('*')))
            print(f"✅ {dir_name}/ ({file_count} items)")
        else:
            print(f"❌ MISSING: {dir_name}/")
    print()
    
    print("=" * 70)
    print("🎉 PROJECT IS NOW CLEAN AND ORGANIZED!")
    print("=" * 70)
    print()
    print("💡 NEXT STEPS:")
    print("   1. Review README.md for project documentation")
    print("   2. Run: python doit.py list (verify pipeline)")
    print("   3. Run: jupyter notebook HMM_Model_Evaluation.ipynb")
    print("   4. Your project is ready for presentation! 🎓")
    print()

if __name__ == "__main__":
    try:
        cleanup()
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        print("Please review and run cleanup manually if needed.")
