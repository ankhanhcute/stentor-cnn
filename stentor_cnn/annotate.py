import h5py 
import sys 

path = sys.argv[1]

f = h5py.File(path, 'r')
print("keys", list(f.keys()))
for k in f.keys():
    print(k, f[k].shape)
    
f.close()