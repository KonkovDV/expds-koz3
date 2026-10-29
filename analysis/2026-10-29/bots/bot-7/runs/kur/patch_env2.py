import sys; f=sys.argv[1]; s=open(f).read()
s=s.replace("        if VARS_ENV and vn not in VARS_ENV.split(','): continue\n","")
s=s.replace("    for vn,v in VAR.items():\n","    for vn,v in VAR.items():\n        if VARS_ENV and vn not in VARS_ENV.split(','): continue\n")
open(f,'w').write(s)
