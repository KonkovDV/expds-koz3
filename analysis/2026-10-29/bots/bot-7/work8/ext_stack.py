import json,sys
SEP='{"type": "Feature"'
out=open('stack_101_103.jsonl','w',encoding='utf-8'); buf=''; n=k=0
with open('/workspace/rerank/stack_arch.geojson',encoding='utf-8') as fh:
    while True:
        ch=fh.read(1<<22)
        buf+=ch
        parts=buf.split(SEP)
        buf=parts.pop() if ch else ''
        for p in parts + ([buf] if not ch else []):
            if not p.startswith(','): pass
            n+=1
            if '101_Гн' in p or '103_Гн' in p:
                s=(SEP+p).rstrip().rstrip(',').rstrip(']}').rstrip()
                # trim trailing separators
                while s and s[-1] in ', \n': s=s[:-1]
                try: o=json.loads(s)
                except Exception:
                    o,_=json.JSONDecoder().raw_decode(SEP+p)
                out.write(json.dumps(o,ensure_ascii=False)+'\n'); k+=1
        if not ch: break
print(n,k)
