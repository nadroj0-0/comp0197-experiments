import subprocess

scripts = ["run_data.py", "run_train_all.py", "run_evaluate_all.py"]

for script in scripts:
    print(f"\n🚀 Executing {script}...")
    result = subprocess.run(["python", script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error in {script}. Pipeline halted.")
        break

print("\n🏆 Full M5 Pipeline Complete! Check outputs/model_comparison.csv")