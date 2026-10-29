import resource,atexit;atexit.register(lambda: print("maxrss_MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024, file=__import__("sys").stderr))
