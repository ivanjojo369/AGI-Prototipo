import subprocess
import datetime
import os
import json

LOG_DIR = "logs/planner/"
os.makedirs(LOG_DIR, exist_ok=True)

def run_script(script_name):
    print(f"▶️ Ejecutando: {script_name}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    return {
        "script": script_name,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    report = {
        "timestamp": timestamp,
        "results": []
    }

    scripts = ["auto_benchmark.py", "stress_test.py"]
    for script in scripts:
        if os.path.exists(script):
            report["results"].append(run_script(script))
        else:
            print(f"⚠️ Script no encontrado: {script}")

    filename = os.path.join(LOG_DIR, f"planner_report_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"📄 Reporte del Planner guardado en: {filename}")

if __name__ == "__main__":
    main()
