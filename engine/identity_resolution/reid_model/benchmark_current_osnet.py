"""Existing encoder validation baseline only. Locked test is not scored here."""
import csv,json,sys
from pathlib import Path
import cv2
import numpy as np
from evaluation import evaluate

HERE=Path(__file__).resolve().parent
PURE=HERE.parent
sys.path.insert(0,str(PURE/'当前方案源码/footballs-main/engine/tracking'))
from tracking_core import SportsOSNetReIDExtractor
rows=list(csv.DictReader((HERE/'data/manifest.csv').open(encoding='utf-8')))
for r in rows:
    for k in ['pid','algorithm_id','source_local_id','frame','kit']:r[k]=int(r[k])
gallery=[r for r in rows if r['enrolment_gallery']=='True']
query=[r for r in rows if r['split']=='val']
chosen=gallery+query
crops=[cv2.imdecode(np.fromfile(HERE/'data'/r['image'],np.uint8),1) for r in chosen]
root=PURE/'当前方案源码/sports_osnet'
model=SportsOSNetReIDExtractor(root,root/'checkpoints/sports_model.pth.tar-60',batch_size=32)
features=model(crops)
report=evaluate(features[len(gallery):],features[:len(gallery)],query,gallery,calibrate=True)
report['note']='Existing OSNet participated in tracking/pseudo-label creation; human groups reviewed, single-clip development validation only.'
(HERE/'osnet_validation_baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['image_retrieval','cross_algorithm_retrieval','track_rank1_macro','same_kit_FAR','TAR']},indent=2))
