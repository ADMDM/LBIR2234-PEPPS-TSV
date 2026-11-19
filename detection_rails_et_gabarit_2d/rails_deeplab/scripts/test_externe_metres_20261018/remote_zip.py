"""Read selected members of a public ZIP via strict HTTP byte ranges."""
import io,urllib.request,zipfile,json,sys,hashlib
from pathlib import Path

class RemoteZipFile(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.pos=0;self.cache={};self.transferred=0
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=40) as r:self.length=int(r.headers['Content-Length'])
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
  if self.pos<0:raise ValueError('negative position')
  return self.pos
 def read(self,n=-1):
  n=self.length-self.pos if n<0 else min(n,self.length-self.pos)
  if not n:return b''
  if n>100_000_000:raise ValueError('Refusing unbounded archive download')
  key=(self.pos,n)
  if key not in self.cache:
   req=urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
   with urllib.request.urlopen(req,timeout=60) as r:
    if r.status!=206:raise RuntimeError('Server must support byte ranges; no full download fallback')
    b=r.read(n+1)
   if len(b)!=n:raise RuntimeError('Wrong range size')
   self.transferred+=n
   if n<10_000_000:self.cache[key]=b
  else:b=self.cache[key]
  self.pos+=n;return b

def download_small(url,path):
 p=Path(path)
 if p.exists():return
 with urllib.request.urlopen(url,timeout=60) as r:
  size=int(r.headers.get('Content-Length',0))
  if size>10_000_000:raise ValueError('Not a small file')
  b=r.read(10_000_001)
 if len(b)>10_000_000:raise ValueError('Exceeded bound')
 p.write_bytes(b)

if __name__=='__main__':
 out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
 base='https://www.mrt.kit.edu/railbench_data/'
 for f,name in [('object_and_rail/annotations.zip','annotations.zip'),('odometry/calibration.zip','calibration.zip'),('odometry/gt_poses.zip','gt_poses.zip')]:download_small(base+f,out/name)
 for f,name in [('object_and_rail/images.zip','IMAGES_ZIP_INDEX.json'),('odometry/scenes.zip','SCENES_ZIP_INDEX.json')]:
  remote=RemoteZipFile(base+f)
  with zipfile.ZipFile(remote) as z:
   a=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,crc=i.CRC) for i in z.infolist()]
  (out/name).write_text(json.dumps(a,indent=2));print(name,len(a),'entries, transferred',remote.transferred,flush=True)
 for n in ['annotations.zip','calibration.zip','gt_poses.zip']:
  with zipfile.ZipFile(out/n) as z:print(n,z.namelist()[:12],flush=True)
