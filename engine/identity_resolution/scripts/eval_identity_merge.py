"""Score second-stage mapping against frozen human identity groups."""
import json
import pickle
from collections import defaultdict
from pathlib import Path


def load_mapping(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8'))
    return {int(k):int(v) for k,v in raw.items()}


def evaluate(audit_path, mapping_path, groups_path):
    audit=pickle.loads(Path(audit_path).read_bytes())
    mapping=load_mapping(mapping_path)
    groups=json.loads(Path(groups_path).read_text(encoding='utf-8'))['groups']
    skip={n['output_id'] for n in audit['nodes'] if n.get('quarantine')}
    present=set()
    frames=defaultdict(int)
    for n in audit['nodes']:
        if n.get('quarantine'):continue
        present.add(n['output_id'])
        frames[n['output_id']]+=len(n['rows'])
    labeled={i for ids in groups.values() for i in ids}
    report={'people':[], 'complete_people':0, 'split_people':0, 'contaminated_people':0, 'cross_person_components':[]}
    used=set()
    for name, ids in groups.items():
        live=[i for i in ids if i in present]
        comps=sorted({mapping.get(i,i) for i in live})
        members=defaultdict(list)
        for i in live:members[mapping.get(i,i)].append(i)
        foreign=[]
        for c in comps:
            others=[i for i,root in mapping.items() if root==c and i in present and i not in ids and i in labeled]
            if others:foreign.append({'component':c,'other_ids':sorted(others)})
        complete=bool(live) and len(comps)==1 and not foreign
        item=dict(name=name, expected_ids=ids, present_ids=live, missing_ids=[i for i in ids if i not in present],
                  components=[{'id':c,'members':sorted(members[c]),'frames':sum(frames[i] for i in members[c])} for c in comps],
                  foreign=foreign, complete=complete, split=len(comps)>1, contaminated=bool(foreign))
        report['people'].append(item)
        report['complete_people']+=int(complete)
        report['split_people']+=int(len(comps)>1)
        report['contaminated_people']+=int(bool(foreign))
        used.update(live)
        if foreign:report['cross_person_components'].extend(foreign)
    leftover=sorted(i for i in present if i not in labeled)
    leftover_comp=defaultdict(list)
    for i in leftover:leftover_comp[mapping.get(i,i)].append(i)
    report.update(people_total=len(groups), unlabeled_present=len(leftover),
                  unlabeled_components={str(k):v for k,v in leftover_comp.items()},
                  all_nine_complete=report['complete_people']==len(groups) and report['contaminated_people']==0)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True);p.add_argument('--groups',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();r=evaluate(a.audit,a.mapping,a.groups)
    text=json.dumps(r,ensure_ascii=False,indent=2)
    if a.output:a.output.write_text(text,encoding='utf-8')
    print(text)
