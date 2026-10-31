import collections, sys, re
inc=collections.Counter(); selfc=collections.Counter(); tot=0
for ln in open(sys.argv[1]):
    st,n=ln.rsplit(' ',1); n=int(n); tot+=n
    fr=[re.sub(r' \(/.*?([^/]+:\d+)\)', r' (\1)', f) for f in st.split(';') if f]
    names=set(re.sub(r':\d+\)$',')',f) for f in fr)
    for k in names: inc[k]+=n
    selfc[fr[-1] if fr else "?"]+=n
print("samples",tot)
print("== inclusive"); [print(f"{100*v/tot:5.1f}% {k[:120]}") for k,v in inc.most_common(int(sys.argv[2]))]
print("== self"); [print(f"{100*v/tot:5.1f}% {k[:120]}") for k,v in selfc.most_common(int(sys.argv[3]))]
