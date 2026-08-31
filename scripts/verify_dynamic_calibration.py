from __future__ import annotations

import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.app import app


def main() -> None:
    client = TestClient(app)
    r = client.post('/api/projects', data={'name':'dynamic-calibration-verification','home_team':'A','away_team':'B'})
    if r.status_code != 200:
        raise RuntimeError(r.text)
    pid = r.json()['id']
    try:
        src = ROOT/'demo_data'/'reference_match'/'source_preview.mp4'
        with src.open('rb') as f:
            r = client.post(f'/api/projects/{pid}/video', files={'video':(src.name,f,'video/mp4')})
        if r.status_code != 200:
            raise RuntimeError(r.text)
        p = r.json(); w=p['video']['width']; h=p['video']['height']; n=p['video']['frame_count']
        # Geometry smoke points: internally self-consistent so the independent scale
        # check tests the workflow, not the real field scale of the preview clip.
        image_points=[[100,100],[w-100,100],[w-100,h-100],[100,h-100]]
        world_points=[[0,0],[45,0],[45,25],[0,25]]
        validations=[{'name':'horizontal_scale_smoke','p1':[100,h/2],'p2':[w-100,h/2],'length_m':45.0}]
        for frame in [0, min(300,n-1)]:
            payload={'frame_index':frame,'image_points':image_points,'world_points':world_points,'validation_segments':validations,'field_length_m':45,'field_width_m':25,'tolerance_m':0.5}
            r=client.post(f'/api/projects/{pid}/calibration/anchors',json=payload)
            if r.status_code != 200 or not any(a.get('frame_index')==frame and a.get('passed') for a in r.json().get('anchors',[])):
                raise RuntimeError(f'anchor failed {frame}: {r.text}')
        r=client.post(f'/api/projects/{pid}/calibration/expand')
        if r.status_code != 200:
            raise RuntimeError(r.text)
        deadline=time.time()+120
        while time.time()<deadline:
            p=client.get(f'/api/projects/{pid}').json(); st=p['calibration']['status']
            if st in {'ready','failed'}:
                break
            time.sleep(.5)
        p=client.get(f'/api/projects/{pid}').json(); c=p['calibration']
        if c['status']!='ready':
            raise RuntimeError(f"dynamic calibration failed: {c.get('message')}")
        v=c.get('validation') or {}
        if float(v.get('accepted_ratio') or 0)<0.8:
            raise RuntimeError(f'coverage too low: {v}')
        print('Dynamic calibration verification: PASS')
        print('  anchors:', len(c.get('anchors') or []))
        print('  accepted_ratio:', v.get('accepted_ratio'))
        print('  accepted_frames:', v.get('accepted_frames'), '/', v.get('total_frames'))
        print('NOTE: point geometry is a workflow smoke test, not a metric-accuracy claim for the preview clip.')
    finally:
        client.delete(f'/api/projects/{pid}')


if __name__=='__main__':
    main()
