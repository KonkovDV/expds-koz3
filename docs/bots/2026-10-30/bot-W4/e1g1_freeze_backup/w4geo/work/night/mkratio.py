from rx import *
P_=["probs_n40r8","probs_n40r6","probs_n40r10","probs_aug","probs_gw1d","probs_gw3d"]
e=np.mean([np.load(W+f"night/s1_{p}.npy") for p in P_],0); b=np.load(W+"night/s1_probs_n40r8.npy")
np.save(W+"night/ratio_ens6.npy",(e+1e-3)/(b+1e-3)); np.save(W+"night/conf_E1.npy",df.conf.astype(float))
