"""Bounded, offline DM0.5 inference with traceable recorded input.

This is an inference integration check. It never counts an input demonstration
as a DSR rollout, skill clip, benchmark success, or embodied task execution.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,signal,sys,threading,time,traceback
from datetime import datetime,timezone
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--observation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--diffusion-steps',type=int,default=10);parser.add_argument('--memory-gib',type=float,default=40);parser.add_argument('--timeout-seconds',type=int,default=600);parser.add_argument('--gpu',type=int);args=parser.parse_args()
if args.memory_gib <= 0 or args.timeout_seconds <= 0 or args.diffusion_steps <= 0: parser.error('budgets and diffusion steps must be positive')
if args.gpu is not None and args.gpu < 0: parser.error('gpu must be nonnegative')
root=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES']='' if args.gpu is None else str(args.gpu);os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(root/'.cache/dm05/huggingface');os.environ['XDG_CACHE_HOME']=str(root/'.cache/dm05/xdg')
for path in [root/'src',root/'vendor',root/'vendor/RoboDojo',root/'vendor/XPolicyLab/policy/OpenDM/opendm']:
 sys.path.insert(0,str(path))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def save():
 result['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
 (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
result={'schema_version':1,'started_at':datetime.now(timezone.utc).isoformat(),'scope':'real released checkpoint inference on pre-recorded demonstration; no new embodied rollout','status':'initializing','device':'cpu' if args.gpu is None else 'cuda:0','physical_gpu':args.gpu,'gpu_hours':0,'api_calls':0,'diffusion_steps':args.diffusion_steps,'seed':0,'checkpoint':str(args.checkpoint.resolve()),'input_file_sha256':sha(args.observation),'code_hashes':{str(p.relative_to(root)):sha(p) for p in [root/'tools/verify_dm05_inference.py',root/'vendor/XPolicyLab/policy/OpenDM/model.py',root/'vendor/XPolicyLab/policy/OpenDM/scripts/robodojo_dm05_history.py',root/'vendor/XPolicyLab/policy/OpenDM/opendm/opendm/model/dm05/dm05_arch.py',root/'vendor/XPolicyLab/policy/OpenDM/opendm/opendm/data/inference_images.py']}}
save();start=time.monotonic()
def budget_watchdog():
 while True:
  time.sleep(0.25)
  rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/(1024**3)
  elapsed=time.monotonic()-start
  if rss > args.memory_gib or elapsed > args.timeout_seconds:
   result.update(status='budget_exceeded',rss_gib=rss,total_seconds=elapsed,memory_limit_gib=args.memory_gib,timeout_seconds=args.timeout_seconds)
   save()
   os._exit(137)
threading.Thread(target=budget_watchdog,daemon=True).start()
try:
 import numpy as np,torch,transformers
 if args.gpu is not None and not torch.cuda.is_available():raise RuntimeError('Requested GPU is unavailable; CPU fallback is forbidden')
 if args.gpu is not None:result['gpu_name']=torch.cuda.get_device_name(0)
 from XPolicyLab.policy.OpenDM.model import Model
 from XPolicyLab.process_data import pack_robot_state,unpack_robot_state
 torch.set_num_threads(8);torch.set_num_interop_threads(2);torch.manual_seed(0)
 result['versions']={'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version};save()
 weight=args.checkpoint/'model.safetensors';stat=weight.stat();stamp=args.checkpoint/'.dsr_verified.json';cached=json.loads(stamp.read_text()) if stamp.exists() else {}
 result['checkpoint_identity']={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256_from_prior_verification':cached.get('sha256'),'prior_verification_matches_current_size_mtime':cached.get('actual_bytes')==stat.st_size and cached.get('mtime_ns')==stat.st_mtime_ns,'prior_verified_at':cached.get('verified_at'),'config_sha256':sha(args.checkpoint/'config.json'),'norm_stats_sha256':sha(args.checkpoint/'norm_stats.json')}
 cfg={'model_path':str(args.checkpoint.resolve()),'norm_stats_path':str((args.checkpoint/'norm_stats.json').resolve()),'experiment_path':str(root/'vendor/XPolicyLab/policy/OpenDM/scripts/robodojo_dm05_history.py'),'action_type':'joint','env_cfg_type':'arx_x5','action_dim':14,'action_chunk_size':50,'action_steps':25,'history_slots':20,'history_fps':1.,'runtime_fps':25.,'history_tokens_per_slot':16,'bf16':False,'liger_kernel':False,'llm_attn_implementation':'sdpa','vision_attn_implementation':'sdpa','action_attn_implementation':'sdpa','diffusion_steps':args.diffusion_steps,'diffusion_noise_seed':0}
 with np.load(args.observation,allow_pickle=False) as f:
  state=np.asarray(f['state'],dtype=np.float32);instruction=str(f['instruction']);images={k:np.asarray(f[k]) for k in ['cam_high','cam_left_wrist','cam_right_wrist']}
 result['input']={'instruction':instruction,'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'state_dim':len(state),'camera_rgb_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()},'camera_shapes':{k:list(v.shape) for k,v in images.items()}};save()
 cfg['prompt']=instruction;model=Model(cfg);result['load_seconds']=time.monotonic()-start;save()
 dims={'arm_dim':[6,6],'ee_dim':[1,1]};obs={'instruction':instruction,'vision':images,'state':unpack_robot_state(state,'joint',dims),'env_idx':0}
 model.update_obs(obs);infer_start=time.monotonic();actions=model.get_action();result['inference_seconds']=time.monotonic()-infer_start
 matrix=np.stack([pack_robot_state({'state':a},'joint',dims) for a in actions]).astype(np.float32)
 if matrix.shape!=(25,14) or not np.isfinite(matrix).all():raise ValueError(f'Invalid action matrix {matrix.shape}')
 np.save(args.output/'actions.npy',matrix);result.update({'status':'executed_successfully','action_count':len(matrix),'action_shape':list(matrix.shape),'actions_sha256':sha(args.output/'actions.npy'),'action_min':float(matrix.min()),'action_max':float(matrix.max()),'action_finite':True,'observation_to_action_conversion_valid':True,'parameter_dtype':str(next(model._inference.model.parameters()).dtype),'total_seconds':time.monotonic()-start});model.reset()
except BaseException as e:
 result.update({'status':'failed','error_type':type(e).__name__,'error':str(e),'total_seconds':time.monotonic()-start});(args.output/'traceback.txt').write_text(traceback.format_exc())
finally:
 if args.gpu is not None:
  result['gpu_hours']=(time.monotonic()-start)/3600
  if 'torch' in locals() and torch.cuda.is_available():result['peak_gpu_allocated_gib']=torch.cuda.max_memory_allocated()/(1024**3)
 save();print(json.dumps(result,indent=2))

if result['status'] != 'executed_successfully':
 sys.exit(1)
