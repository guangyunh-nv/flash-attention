import json
import os
import runpy
import sys
from pathlib import Path
import types

root=Path(os.environ['MATRIX_REPO'])
# Select exact source trees without changing the installed package or either ref.
namespace=types.ModuleType('flash_attn');namespace.__path__=[str(root/'flash_attn')];sys.modules['flash_attn']=namespace
namespace=types.ModuleType('flash_attn.cute');namespace.__path__=[str(root/'flash_attn/cute')];sys.modules['flash_attn.cute']=namespace
from flash_attn.cute import interface
metadata=[]
for cls in (interface.FlashAttentionForwardSm100,interface.BlackwellFusedMultiHeadAttentionForward):
 original=cls._setup_attributes
 def setup(self,original=original):
  original(self)
  if os.environ['MATRIX_DEPTH_POLICY']=='matched_max' and os.environ['MATRIX_REVISION']=='base':
   self.kv_stage=int(os.environ['FA_HD256_KV_STAGE'])
  item=dict(kernel=type(self).__name__,q_stage=self.q_stage,kv_stage=self.kv_stage,cluster=self.cluster_shape_mn,clc=self.use_clc_scheduler,persistent=self.is_persistent)
  metadata.append(item);print('specialization='+json.dumps(item),flush=True)
 cls._setup_attributes=setup
out=Path(os.environ['MATRIX_OUTPUT'])
sys.argv=['benchmark_preferred_clusters.py','--gpu',os.environ['MATRIX_GPU'],'--dtype',os.environ['MATRIX_DTYPE'],'--head-dim',os.environ['MATRIX_DIM'],'--mode',os.environ['FA_FWD_CLUSTER_MODE'],'--warmup','32','--repeats','40','--output',str(out)]
if os.environ['MATRIX_DIM']=='256':sys.argv+=['--kv-stages',os.environ['FA_HD256_KV_STAGE']]
runpy.run_path(os.environ['MATRIX_BENCHMARK'],run_name='__main__')
r=json.loads(out.read_text());r.update(revision=os.environ['MATRIX_REVISION'],depth_policy=os.environ['MATRIX_DEPTH_POLICY'],specializations=metadata,commit=os.environ['MATRIX_COMMIT'])
assert len(metadata)==1,metadata
assert metadata[0]['cluster']==((2,1) if os.environ['FA_FWD_CLUSTER_MODE']=='2cta' else (8,1))

spec=metadata[0]
expected_stage=(6 if os.environ['MATRIX_DTYPE']=='bf16' else 16) if os.environ['MATRIX_DIM']=='128' else int(os.environ['FA_HD256_KV_STAGE'])
assert spec['q_stage']==2 and spec['kv_stage']==expected_stage,spec
assert not spec['clc'],spec
assert len(r['samples_ms'])==40 and r['frequency_samples'],r.keys()
out.write_text(json.dumps(r,indent=2)+'\n')
