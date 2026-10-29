import importlib.util,unittest,tempfile,json,zipfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('all_support',ROOT/'scripts/v8_all_support.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AllSupportTests(unittest.TestCase):
 def make_case(self,case,t=1,s=0):
  case.mkdir(parents=True);(case/'audit.json').write_text(json.dumps({'config':{'protocol':'v8_validation_v1','task_id':t,'init_state_id':s},'status':'paused_branch_budget'}));(case/'run.log').write_text('log')
 def test_checkpoint_and_old_style_restore(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.make_case(p/'case');m.checkpoint(p/'case',p/'checkpoint.zip');rows=m.restore(p/'checkpoint.zip',p/'new',ROOT);self.assertEqual(rows[0]['task_id'],1)
 def test_all_export_restore_two_cases(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.make_case(p/'old/task-1-init-0/case');self.make_case(p/'old/task-2-init-1/case',2,1)
   m.export_all(p/'old',p/'all.zip');rows=m.restore(p/'all.zip',p/'new',ROOT);self.assertEqual(len(rows),2)
 def test_existing_case_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.make_case(p/'case');m.checkpoint(p/'case',p/'c.zip');self.make_case(p/'new/task-1-init-0/case')
   with self.assertRaises(ValueError):m.restore(p/'c.zip',p/'new',ROOT)
 def test_traversal_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with zipfile.ZipFile(p/'bad.zip','w') as z:z.writestr('../bad','bad')
   with self.assertRaises(ValueError):m.restore(p/'bad.zip',p/'new',ROOT)
 def test_incomplete_case_not_reported_complete(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.make_case(p/'case');self.assertIsNone(m.complete_case(p/'case',1,0,ROOT))
if __name__=='__main__':unittest.main()
