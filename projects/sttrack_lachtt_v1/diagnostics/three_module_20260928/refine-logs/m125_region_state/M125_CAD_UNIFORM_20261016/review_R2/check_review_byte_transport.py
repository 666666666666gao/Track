"""Reproduce launcher text-versus-byte mismatch without any remote operation."""
import ast,base64,hashlib,json,zlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
invoker=BASE/'M125_CAD_DEPLOYMENT_SOURCE_20261016/invoke_M125_CAD_reviewed_suite.py'
tree=ast.parse(invoker.read_text(encoding='utf-8'))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
remote=next(n.value.func.value.value for n in main.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='remote')
remote_tree=ast.parse(remote)
assert_node=next(n for n in remote_tree.body if isinstance(n,ast.Assert) and 'source_review_sha256' in ast.unparse(n))
rows=[]
original=BASE/'M125_CAD_UNIFORM_SOURCE_AUDIT_20261016'
for name in ['EXPERIMENT_AUDIT.json','EXPERIMENT_AUDIT.md']:
    file=original/name;raw=file.read_bytes();text=file.read_text(encoding='utf-8');encoded=text.encode()
    rows.append(dict(path=str(file),raw_bytes=len(raw),transported_bytes=len(encoded),
        raw_sha256=hashlib.sha256(raw).hexdigest(),transported_sha256=hashlib.sha256(encoded).hexdigest(),
        byte_exact=raw==encoded,CRLF_count=raw.count(b'\r\n'),LF_count=raw.count(b'\n')))
request=dict(source_review_json=(original/'EXPERIMENT_AUDIT.json').read_text(encoding='utf-8'),
    source_review_md=(original/'EXPERIMENT_AUDIT.md').read_text(encoding='utf-8'),
    source_review_sha256=rows[0]['raw_sha256'])
wire=base64.b64encode(zlib.compress(json.dumps(request).encode())).decode()
p=json.loads(zlib.decompress(base64.b64decode(wire)))
try:
    exec(compile(ast.Module(body=[assert_node],type_ignores=[]),str(invoker)+' embedded remote','exec'),dict(hashlib=hashlib,p=p))
    result='accepted'
except AssertionError:
    result='AssertionError'
assert result=='AssertionError' and all(not r['byte_exact'] for r in rows)
receipt=dict(status='reproduced_original_review_byte_transport_mismatch',evaluation_type='simulation_only',fixture=True,
    actual_SSH_calls=0,actual_remote_execution=False,exact_embedded_remote_assertion=ast.unparse(assert_node),
    assertion_result=result,files=rows,source_sha256=hashlib.sha256(invoker.read_bytes()).hexdigest(),
    pipeline='actual local read_text/encode semantics -> json.dumps -> zlib.compress -> base64 -> inverse -> exact remote assertion AST')
(OUT/'review_byte_transport_reproduction.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
print(json.dumps(receipt,indent=2))
