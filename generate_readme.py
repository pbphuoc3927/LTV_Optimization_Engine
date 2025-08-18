import os
from pathlib import Path

def generate_readme(base_dir="."):
    content = []

    # 1. Project Title
    content.append("# Customer Lifetime Value (LTV) Optimization Engine\n")

    # 2. Badge / Status (tùy chọn)
    content.append("![status](https://img.shields.io/badge/status-active-brightgreen)\n")

    # 3. Objective
    content.append("## 🎯 Objective\n")
    content.append("Increase customer LTV by **15–25%** through predictive modeling & targeted interventions.\n")

    # 4. Tech Stack
    content.append("## ⚙️ Tech Stack\n")
    techs = [
        "Python", "pandas", "numpy", "scikit-learn", "xgboost", 
        "matplotlib", "seaborn", "plotly", "sqlalchemy", 
        "pyyaml", "faker", "great-expectations", "MLflow"
    ]
    content.append("- " + "\n- ".join(techs) + "\n")

    # 5. Folder Structure
    content.append("## 📂 Project Structure\n")
    structure = """
```

LTV\_Optimization\_Engine/
│── data/                 # Raw & processed datasets
│── outputs/              # Model results, charts, KPI reports
│── scripts/              # All analysis & automation scripts
│── tests/                # Automated test suite
│── compile\_results.py    # Compile final results into Markdown
│── generate\_readme.py    # Auto-generate README.md

```
"""
    content.append(structure + "\n")

    # 6. Scripts
    content.append("## 🖥️ Scripts\n")
    scripts = {
        "generate_mock_data.py": "Generate synthetic customer data.",
        "analysis_pipeline.py": "Perform LTV modeling, KPI calculation, and insights.",
        "chart_generator.py": "Create charts & visualizations.",
        "compile_results.py": "Aggregate results into a single Markdown report.",
        "generate_readme.py": "Auto-generate this README file.",
        "run_complete_analysis.py": "Run full pipeline end-to-end.",
    }
    for k, v in scripts.items():
        content.append(f"- **{k}** → {v}\n")

    # 7. Outputs
    content.append("\n## 📊 Outputs\n")
    outputs = {
        "ltv_results.csv": "Customer-level predicted LTV values.",
        "kpi_summary.csv": "Key metrics & ROI calculations.",
        "charts/*.png": "Visualizations of LTV distributions, feature importances, etc.",
        "compiled_results.md": "Final consolidated analysis report."
    }
    for k, v in outputs.items():
        content.append(f"- **{k}** → {v}\n")

    # 8. Audience
    content.append("\n## 👥 Target Audience\n")
    content.append("Designed for **C-suite, Board Members, and Marketing Leadership**.\n")

    return "\n".join(content)

def scan_outputs(base_dir="outputs"):
    """Scan outputs folder and return list of generated files."""
    outputs = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), base_dir)
            outputs.append(rel_path)
    return outputs

if __name__ == "__main__":
    readme_content = generate_readme()
    readme_path = Path(r"C:\Users\phuoc\LTV_Optimization_Engine\README.md")
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"✅ README.md generated at {readme_path.resolve()}")