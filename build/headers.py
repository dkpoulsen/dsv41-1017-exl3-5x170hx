import glob, os, shutil
inc = "/usr/local/cuda/include"
os.makedirs(inc, exist_ok=True)
roots = glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/*/include") + \
        glob.glob("/usr/local/lib/python3.12/site-packages/nvidia/*/include")
n = 0
for r in roots:
    for f in glob.glob(os.path.join(r, "*.h")):
        base = os.path.basename(f)
        if not os.path.exists(os.path.join(inc, base)):
            shutil.copy2(f, os.path.join(inc, base)); n += 1
print("copied", n, "headers from", len(roots), "dirs")
for want in ("cusparse.h", "cublas_v2.h", "cusparseLt.h"):
    print(want, "->", os.path.exists(os.path.join(inc, want)))
