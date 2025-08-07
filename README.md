# ascend-ops

Ascend 910A相关的算子实现，配套https://github.com/shirwy/vllm-ascend-910a仓使用

验证过的容器镜像
* `quay.io/ascend/cann:8.1.rc1-910b-openeuler22.03-py3.10`
* `quay.io/ascend/vllm-ascend:v0.9.1rc1`

## Build

```bash
# 先装torch-npu再source对应的环境变量
python3 -m pip install -r requirements-build.txt


source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib

python3 -m pip install -v -e . --no-build-isolation
```
