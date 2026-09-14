"""Visual QA of native algorithm crops in all three splits."""
import csv
from pathlib import Path
import cv2
import numpy as np

p=Path(__file__).resolve().parent
rows=list(csv.DictReader((p/'data/manifest.csv').open(encoding='utf-8')))
dest=p/'data/review';dest.mkdir(exist_ok=True)
strips=[]
for pid in range(9):
    tiles=[]
    for split in ['train','val','test']:
        rr=[r for r in rows if int(r['pid'])==pid and r['split']==split]
        for idx in np.linspace(0,len(rr)-1,4).astype(int):
            r=rr[idx];im=cv2.imdecode(np.fromfile(p/'data'/r['image'],np.uint8),1)
            tile=np.full((180,104,3),245,np.uint8)
            scale=min(100/im.shape[1],145/im.shape[0])
            im=cv2.resize(im,(max(1,int(im.shape[1]*scale)),max(1,int(im.shape[0]*scale))))
            tile[30:30+im.shape[0],:im.shape[1]]=im
            cv2.putText(tile,split+' f'+r['frame'],(1,14),0,.34,(0,0,0),1)
            tiles.append(tile)
    strip=np.hstack(tiles);ok,b=cv2.imencode('.jpg',strip);assert ok;b.tofile(str(dest/f'pid_{pid}.jpg'));strips.append(strip)
ok,b=cv2.imencode('.jpg',np.vstack(strips));assert ok;b.tofile(str(dest/'all_identities_splits.jpg'))
print('review sheets saved')
