import os
basedir = './'
for fn in os.listdir(basedir):
    if 'ped' in fn:
        print(fn)
        os.rename(os.path.join(basedir, fn), os.path.join(basedir, fn.split('_')[-1]))
print('success')
