from feats import *
d,F,O=split(); df=table(F)
score(F,"E1")
for dd in (1.25,1.5,2.0,2.5,3.0):
    for cross,same in ((1,0),(0,1),(1,1)):
        c,k=nms(df,dd,cross,same); score(with_conf(F,c,k),f"nms{dd} x{cross}s{same} drop={(~k).sum()}")
for dd in (1.5,2.0,3.0):
    for s in (0.5,0.7,0.85):
        c,k=nms(df,dd,1,1,soft=s); score(with_conf(F,c),f"soft{dd} x{s}")
