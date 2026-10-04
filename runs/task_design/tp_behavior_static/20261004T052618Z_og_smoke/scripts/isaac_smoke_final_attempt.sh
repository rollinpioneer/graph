V=/home/xushijie2/xsj2_og_env
source $V/bin/activate
unset EXP_PATH CARB_APP_PATH ISAAC_PATH PYTHONPATH
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1
cd /home/xushijie2
echo "== where did the previous run crash (tail of log2)"
grep -a -v -E "^\s*$" /home/xushijie2/xsj2_isaac_smoke2.log | tail -14 | cut -c1-240
echo "== user-space RPMs for libXt / libGLU? (no root)"
mkdir -p /home/xushijie2/xsj2_og_libs && cd /home/xushijie2/xsj2_og_libs
(timeout 120 dnf download --destdir . libXt mesa-libGLU 2>&1 | tail -6) || echo "dnf download failed"
ls *.rpm 2>/dev/null
for r in *.rpm; do [ -f "$r" ] && rpm2cpio "$r" | cpio -idm 2>/dev/null; done
find . -name "libXt.so*" -o -name "libGLU.so*" | head
cd /home/xushijie2
echo "== retry: preload torch's own libgomp + user-space libs"
TG=$(ls $V/lib/python3.11/site-packages/torch/lib/libgomp*.so.1 | head -1)
L=/home/xushijie2/xsj2_og_libs
export LD_LIBRARY_PATH=$L/usr/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$TG
echo "preload: $TG"
timeout 900 python /home/xushijie2/xsj2_isaac_smoke.py > /home/xushijie2/xsj2_isaac_smoke3.log 2>&1
echo EXIT=$?
grep -a "ISAAC_SMOKE_RESULT" /home/xushijie2/xsj2_isaac_smoke3.log | cut -c1-1200
echo "== distinct errors"
grep -a "\[Error\]" /home/xushijie2/xsj2_isaac_smoke3.log | sed -E 's/^[^ ]+ \[[^]]*\] //' | sed -E 's#/home/xushijie2/xsj2_og_env/lib/python3.11/site-packages/isaacsim/##g' | cut -c1-200 | sort | uniq -c | sort -rn | head -8
grep -a -v -E "^\s*$" /home/xushijie2/xsj2_isaac_smoke3.log | tail -5 | cut -c1-240
