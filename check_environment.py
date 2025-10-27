"""
环境检查脚本
"""
import sys

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError as e:
        print(f"❌ {package_name} not installed or broken: {e}")
        return False

def main():
    print("="*80)
    print("Environment Check")
    print("="*80)
    
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    print("\n" + "="*80)
    print("Checking required packages...")
    print("="*80)
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('xarray', 'xarray'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('torch', 'torch'),
        ('dask', 'dask'),
    ]
    
    results = []
    for pkg_name, import_name in required_packages:
        result = check_package(pkg_name, import_name)
        results.append((pkg_name, result))
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    n_success = sum(1 for _, r in results if r)
    n_total = len(results)
    
    print(f"\n{n_success}/{n_total} packages OK")
    
    if n_success < n_total:
        print("\n⚠️  Missing packages detected!")
        print("\nTo fix, run:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        for pkg_name, result in results:
            if not result:
                print(f"  pip install {pkg_name}")
        
        # PyTorch特殊说明
        if not any(r for p, r in results if p == 'torch'):
            print("\n⚠️  PyTorch needs special attention:")
            print("  Visit: https://pytorch.org/get-started/locally/")
            print("  For CPU-only (recommended for testing):")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
        
        return 1
    else:
        print("\n✓ All packages installed correctly!")
        print("\nYou can now run:")
        print("  python test_pipeline.py")
        return 0

if __name__ == '__main__':
    sys.exit(main())

