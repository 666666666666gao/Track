from pathlib import Path
import ast,base64,hashlib,zlib
R=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902')
A=R/'.aris/m122_full_causal_20261007'
D=R/'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
source_path=D/'analyze_m124_epoch01_cdtb_saved_trajectory.py'
source=source_path.read_bytes();source_sha=hashlib.sha256(source).hexdigest()
ast.parse(source)
remote_path='/root/autodl-tmp/m124_epoch01_cdtb_saved_trajectory_diagnostic_20261012.py'
bootstrap="from pathlib import Path\nimport base64,hashlib,runpy,zlib\nsource=zlib.decompress(base64.b64decode("+repr(base64.b64encode(zlib.compress(source)).decode())+"))\nassert hashlib.sha256(source).hexdigest()=="+repr(source_sha)+"\npath=Path("+repr(remote_path)+")\nassert not path.exists()\npath.write_bytes(source)\nassert hashlib.sha256(path.read_bytes()).hexdigest()=="+repr(source_sha)+"\nrunpy.run_path(str(path),run_name='__main__')\n"
ast.parse(bootstrap)
old=ast.parse((A/'CurrentC_CDTB_recall_diagnostic_native_20261011.py').read_text(encoding='utf-8'))
args=next(ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
command="CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/mplt/bin/python -B -u -c 'import base64,zlib;exec(zlib.decompress(base64.b64decode(\""+base64.b64encode(zlib.compress(bootstrap.encode())).decode()+"\")))'"
args=args[:-1]+[command]
target=A/'M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py'
assert not target.exists()
text='from pathlib import Path\nimport subprocess,sys\nargs='+repr(args)+'\nwith Path('+repr(str(A/'M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json'))+").open('xb') as stream:\n    result=subprocess.run(args,stdout=stream)\nsys.exit(result.returncode)\n"
ast.parse(text);target.write_text(text,encoding='utf-8')
print(dict(status='prepared_CPU_diagnostic_not_executed',source_sha256=source_sha,remote_new_diagnostic_path=remote_path,NN_calls=0,GPU_queries=0,training_progress_queries=0))
