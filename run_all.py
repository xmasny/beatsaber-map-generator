import subprocess
import sys


USE_PARALLEL = True  # Set to False to disable multiprocessing
SKIP_EXISTING = True  # Set to True to skip already processed songs

parallel_flag = ["--parallel"] if USE_PARALLEL else []
skip_flag = ["--skip-existing"] if SKIP_EXISTING else []

# print("🧪 Running validate_metadata.py...")
# result = subprocess.run(["python", "validate_metadata.py"] + parallel_flag)
# if result.returncode != 0:
#     print("❌ Validation failed.")
#     sys.exit(result.returncode)

print("✅ Validation succeeded. 🛠 Now processing notes...")
result = subprocess.run(["python", "process_notes.py"] + parallel_flag + skip_flag)
if result.returncode != 0:
    print("❌ Note processing failed.")
    sys.exit(result.returncode)

print("🎉 All done!")
