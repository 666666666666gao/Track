from pathlib import Path
import subprocess,time
R=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
while not (R/'controller.exit').exists():time.sleep(240)
assert (R/'controller.exit').read_text().strip()=='0'
subprocess.run(['/root/autodl-tmp/envs/sttrack/bin/python','/root/autodl-tmp/audit_m82_completed_20260909.py'],check=True)
