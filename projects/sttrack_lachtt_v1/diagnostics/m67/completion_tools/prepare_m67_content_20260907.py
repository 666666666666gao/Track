from pathlib import Path
import hashlib
BASE=Path('/root/autodl-tmp');OLD=BASE/'m65_content_counterfactuals_20260907.py';OUT=BASE/'m67_content_counterfactuals_20260907.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='3e125d263862f8f301417423cc25eb06dc2191fbe9d28404e3e8562509ae9bd3'
s=OLD.read_text().replace('sttrack_m65_category_null_support_20260907','sttrack_m67_supervised_semantic_support_20260907')
s=s.replace('m65_content_counterfactuals_20260907.py','m67_content_counterfactuals_20260907.py').replace('audit_m65_completed_20260907.py','audit_m67_completed_20260907.py')
s=s.replace('fc04a897f3d3e246203981a2cb3b83ea50075392e30c0c960c8908eaaeabb93b','2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e')
s=s.replace('09f3f193de81f9cf91518b2c05497bfea30e8ab88db812e3585f6d3618767723','d4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5')
s=s.replace('ef6a3f5f9f7b8635497fc993fded0924e7f37f5b67d14c3fab46fc27b7e596db','1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf')
s=s.replace('M65','M67').replace("'null'","'support'").replace('training/null','training/support').replace('recursive/null','recursive/support').replace('null_recursive_receipt','support_recursive_receipt').replace(' null final',' support final')
OUT.write_text(s);print('CONTENT_SOURCE_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
