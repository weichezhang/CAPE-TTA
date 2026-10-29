import importlib.util,unittest,tempfile,zipfile,json,hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('bundle',Path(__file__).resolve().parents[1]/'src/cape_tta/audit_bundle.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class BundleTests(unittest.TestCase):
 def archive(self,path,*,corrupt=False,escape=False):
  data=json.dumps({'config':{'protocol':'v8_validation_v1','task_id':1,'init_state_id':0},'status':'paused_branch_budget'}).encode()
  with zipfile.ZipFile(path,'w') as z:
   z.writestr('case/audit.json',data)
   z.writestr('case/sha256.json',json.dumps({'audit.json':'0'*64 if corrupt else hashlib.sha256(data).hexdigest()}))
   if escape:z.writestr('../escape','bad')
 def test_valid_restore(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.archive(p/'run.zip');self.assertEqual(m.restore_case_bundle(p/'run.zip',p/'case',task_id=1,init_state_id=0),'paused_branch_budget');self.assertTrue((p/'case/audit.json').is_file())
 def test_corruption_leaves_no_case(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.archive(p/'run.zip',corrupt=True)
   with self.assertRaises(ValueError):m.restore_case_bundle(p/'run.zip',p/'case',task_id=1,init_state_id=0)
   self.assertFalse((p/'case').exists())
 def test_traversal_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.archive(p/'run.zip',escape=True)
   with self.assertRaises(ValueError):m.restore_case_bundle(p/'run.zip',p/'case',task_id=1,init_state_id=0)
 def test_wrong_case_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.archive(p/'run.zip')
   with self.assertRaises(ValueError):m.restore_case_bundle(p/'run.zip',p/'case',task_id=2,init_state_id=0)
 def test_existing_case_not_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.archive(p/'run.zip');(p/'case').mkdir()
   with self.assertRaises(ValueError):m.restore_case_bundle(p/'run.zip',p/'case',task_id=1,init_state_id=0)
if __name__=='__main__':unittest.main()
