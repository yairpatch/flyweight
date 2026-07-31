import struct,sys,collections,re
p=sys.argv[1]
f=open(p,'rb')
magic,ver,ntensor,nkv=struct.unpack('<IIQQ',f.read(24))
def rd_str():
    n,=struct.unpack('<Q',f.read(8)); return f.read(n).decode('utf-8',errors='replace')
FMT={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}
SZ={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def rd_val(t):
    if t==8: return rd_str()
    if t==9:
        et,=struct.unpack('<I',f.read(4)); n,=struct.unpack('<Q',f.read(8))
        if et==8: return [rd_str() for _ in range(n)]
        if et==9: return ['<nested>']
        return list(struct.unpack('<%d%s'%(n,FMT[et]),f.read(n*SZ[et])))
    return struct.unpack('<'+FMT[t],f.read(SZ[t]))[0]
kv={}
for _ in range(nkv):
    k=rd_str(); t,=struct.unpack('<I',f.read(4)); kv[k]=rd_val(t)
tensors=[]
for _ in range(ntensor):
    name=rd_str(); nd,=struct.unpack('<I',f.read(4))
    dims=list(struct.unpack('<%dQ'%nd,f.read(8*nd)))
    ty,=struct.unpack('<I',f.read(4)); off,=struct.unpack('<Q',f.read(8))
    tensors.append((name,dims,ty,off))
for k,v in kv.items():
    s=str(v)
    if len(s)>110: s=s[:110]+'...'
    print('KV',k,'=',s)
print('--- tensors ---')
agg=collections.OrderedDict()
for name,dims,ty,off in tensors:
    n=re.sub(r'\.\d+\.','.N.',name)
    key=(n,ty,tuple(dims))
    agg[key]=agg.get(key,0)+1
for (n,ty,dims),c in agg.items():
    print(f'{n:45s} type={ty:3d} dims={dims} x{c}')
