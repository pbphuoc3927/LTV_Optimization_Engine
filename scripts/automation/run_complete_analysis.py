import subprocess

print("🚀 Running complete analysis workflow")

steps = [
    r"C:\Users\phuoc\LTV_Optimization_Engine\scripts\analysis\analysis_pipeline.py",
    r"C:\Users\phuoc\LTV_Optimization_Engine\scripts\analysis\chart_generator.py",
    r"C:\Users\phuoc\LTV_Optimization_Engine\scripts\analysis\generate_mock_data.py"
]

for step in steps:
    print(f"▶️ Executing: {step}")
    subprocess.run(step, shell=True, check=True)

print("✅ Workflow finished successfully")