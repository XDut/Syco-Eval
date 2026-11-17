"""Local-only inference adapter. No hosted API requests, secrets, or HTTP inference server."""
class ModelClient:
    def __init__(self, model='Qwen/Qwen2.5-0.5B-Instruct',max_new_tokens=96):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError('Install requirements-local.txt for local model inference.') from exc
        self.model=model
        self.max_new_tokens=max_new_tokens
        kwargs={'device_map':'auto'} if torch.cuda.is_available() else {'device':-1}
        self.pipe=pipeline('text-generation',model=model,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,**kwargs)
    def generate(self,system,user,temperature=0.):
        settings={'max_new_tokens':self.max_new_tokens,'do_sample':temperature>0,'return_full_text':False}
        if temperature>0:settings['temperature']=temperature
        out=self.pipe([{'role':'system','content':system},{'role':'user','content':user}],**settings)
        v=out[0]['generated_text']
        return v[-1]['content'] if isinstance(v,list) else str(v)
