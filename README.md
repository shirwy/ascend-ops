# ascend910a-extras

910A相关的算子实现

验证过的容器镜像
* `quay.io/ascend/cann:8.1.rc1-910b-openeuler22.03-py3.10`

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib

python3 -m pip install -v -e . --extra-index https://mirrors.huaweicloud.com/ascend/repos/pypi
```