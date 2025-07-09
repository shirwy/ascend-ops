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

## 添加新算子

首先写一个`op_def.json`，例如
```json
[
  {
      "op": "SwiGluEx",
      "input_desc": [
          {
              "name": "x",
              "param_type": "required",
              "format": [
                  "ND"
              ],
              "type": [
                  "fp16"
              ]
          }
      ],
      "output_desc": [
          {
              "name": "y",
              "param_type": "required",
              "format": [
                  "ND"
              ],
              "type": [
                  "fp16"
              ]
          }
      ]
  }
]
```

然后使用`msOpGen`在已有基础上生成新算子框架
```bash
# 注意-m需要是1
msopgen gen -m 1 -i op_def.json -f pytorch -c ai_core-ascend910 -lan cpp -out ./csrc/opdev
```

在生成的代码框架里写计算逻辑，然后重新编译整个项目即可
