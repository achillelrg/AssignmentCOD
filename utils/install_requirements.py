import os
import sys
import subprocess


def install_requirements(requirements_file="requirements.txt"):
    """
    Installs packages from requirements.txt one by one with clear progress logging.
    """
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        sys.exit(1)

    with open(requirements_file, 'r') as f:
        # Filter empty lines and comments
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    total = len(packages)
    print(f"\n--- Installing {total} Dependencies ---")

    for i, package in enumerate(packages, 1):
        # Handle version specifiers for display (e.g. numpy>=1.20 -> numpy)
        pkg_name = package.split('>')[0].split('<')[0].split('=')[0]
        
        print(f"[{i}/{total}] Checking {pkg_name}...", end=" ", flush=True)
        
        try:
            # Check if installed
            subprocess.check_call([sys.executable, "-m", "pip", "show", pkg_name], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Already installed.")
        except subprocess.CalledProcessError:
            print(f"⬇ Downloading & Installing...", end=" ", flush=True)
            try:
                # Install
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ Done.")
            except subprocess.CalledProcessError:
                print(f"❌ FAILED.")
                print(f"Create a manual issue for: {package}")
                sys.exit(1)

    print("\n✅ All dependencies are ready.\n")

if __name__ == "__main__":
    # Allow optional path arg, default to plain requirements.txt in root
    req_path = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    install_requirements(req_path)
