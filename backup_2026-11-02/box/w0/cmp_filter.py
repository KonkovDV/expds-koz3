import ijson, sys, itertools
a=ijson.items(open(sys.argv[1],'rb'),'features.item',use_float=True)
b=ijson.items(open(sys.argv[2],'rb'),'features.item',use_float=True)
n=0
for x,y in itertools.zip_longest(a,b):
    if x!=y: print('DIFF at',n); sys.exit(1)
    n+=1
print('EQUAL features',n)
