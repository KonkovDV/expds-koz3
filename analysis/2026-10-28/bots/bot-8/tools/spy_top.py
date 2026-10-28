import collections, sys
c=collections.Counter(); tot=0
key=sys.argv[2] if len(sys.argv)>2 else 'build_curves'
for ln in open(sys.argv[1]):
    st,n=ln.rsplit(' ',1); n=int(n)
    fr=st.split(';')
    if not any(key in f for f in fr): continue
    tot+=n
    c[fr[-1] if fr[-1] else fr[-2]+' <native>']+=n
print("samples", tot)
for k,v in c.most_common(int(sys.argv[3]) if len(sys.argv)>3 else 20): print(f"{100*v/tot:5.1f}% {k[:150]}")
