import json,glob,os,collections,sys
TH2={'kurgany_povrezhdennye':0.514733,'kurgany_tselye':0.701621}; THA2={'kurgany_povrezhdennye':0.501645,'kurgany_tselye':0.645505}
TH1={'kurgany_povrezhdennye':0.27822,'kurgany_tselye':0.3696}
def scan(d):
    c=collections.Counter()
    for fp in glob.glob(d+'/*.jsonl'):
        for ln in open(fp,encoding='utf-8'):
            p=json.loads(ln)['properties']; g=p.get('threshold_group') or p.get('markup_type'); k=p.get('class_name'); cf=float(p.get('confidence',0))
            c[('n',g,k)]+=1 if g in('sp_or','sp_focal','sp_focal2') else 0
            if g=='sp_focal2' and k in TH2:
                c[('sp2_ge_new',k)]+= cf>=TH2[k]; c[('sp2_ge_Afloor',k)]+= cf>=THA2[k]
            if g in('sp_focal','sp_or') and k in TH1: c[('sp1_ge_new',g,k)]+= cf>=TH1[k]
    return {' '.join(map(str,k)):v for k,v in sorted(c.items(),key=str) if v}
out={d:scan(d) for d in sys.argv[1:]}
print(json.dumps(out,ensure_ascii=False,indent=1))
