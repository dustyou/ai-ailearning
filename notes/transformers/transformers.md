教学视频: https://www.bilibili.com/video/BV1ma4y1g791/?vd_source=a2f6ab85bc9c65007fe55aa705019af5

github: https://github.com/zyds/transformers-code



# 01-Getting Started

## conda

### 下载conda并安装

清华镜像源

https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

下载这个进行安装

安装路径不要有中文

[Miniconda3-py39_22.11.1-1-Windows-x86_64.exe](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_22.11.1-1-Windows-x86_64.exe)

配置conda路径到环境变量 PATH

配置conda路径\Scripts 到环境变量 PATH

这样连python, pip一起安装了



### 初始化

```shell
conda create -n transformers_01 python=3.9

# 激活环境
conda init
conda activate transformers_01
conda deactivate
```

看到命令行前面出现(transformers), 表示激活成功了

### 设置国内镜像源

清华镜像源首页

https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

设置默认镜像源为清华镜像源

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```



### 安装cuda

查看系统显卡信息

桌面 -> 右键 -> NVIDIA控制面板 -> 左下角 系统信息

显卡最高支持cuda 9.2

![image-20240204003053286](image/image-20240204003053286.png)

到pytorch官网

https://pytorch.org/

找到历史版本

找到支持cuda 9.2的

![image-20240204003641459](image/image-20240204003641459.png)

```
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```



装包

```shell
pip install datasets evaluate peft accelerate gradio optimum sentencepiece
pip install jupyterlab scikit-learn pandas matplotlib tensorboard nltk rouge

```



hosts修改

```
185.199.108.133 raw.githubusercontent.com
185.199.109.133 raw.githubusercontent.com
185.199.110.133 raw.githubusercontent.com
185.199.111.133 raw.githubusercontent.com
2606:50c0:8000:154 raw.githubusercontent.com
2606:50c0:8001:154 raw.githubusercontent.com
2606:50c0:8002:154 raw.githubusercontent.com
2606:50c0:8003:154 raw.githubusercontent.com
```



### pycharm使用和配置jupyter

https://zhuanlan.zhihu.com/p/667528990?utm_id=0



### conda命令

https://www.jb51.net/python/301666gz8.htm#_label3

```
conda create -n transformers_01 python=3.9

# 激活环境
conda init
conda activate transformers_01
conda deactivate
# 删除环境
conda remove -n test --all
# clone环境
conda create -n NewName --clone OldName #把环境 OldName 重命名成 NewName

conda config --show


conda config --remove envs_dirs  D:\ProgramFiles\condadata\envs
```

修改配置, 解决c盘占满问题

http://www.taodudu.cc/news/show-2776654.html?action=onClick



配置文件位置: .condarc





## 设置transformers从镜像网站下载模型

参考: https://blog.csdn.net/popboy29/article/details/135512259

在from transformers import pipeline之前加下面的代码

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```



注意, 一定要在 from transformers import pipeline之前加, 否则不生效

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
hf_endpoint = os.getenv("HF_ENDPOINT")
print(hf_endpoint)
from transformers import pipeline

pipe = pipeline("text-classification")
pipe(["very good!", "vary bad!"])
```
# windows创建文件夹链接
```cmd
mklink /D <Link> <Target>

mklink /D C:\Users\Michael\AppData\Roaming\Python1 E:\C_Cache\Users\Michael\AppData\Roaming\Python

```

# 使用huggingface-cli下载模型
1 Hugging Face
使用 Hugging Face 官方提供的 huggingface-cli 命令行工具。安装依赖:

pip install -U huggingface_hub
-U 意思为安装最新版本，若已经安装则更新至最新版本

然后新建 python 文件，填入以下代码，运行即可。

resume-download：断点续下
local-dir：本地存储路径。（linux 环境下需要填写绝对路径）


## 下载模型
```angular2html
import os
os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')

huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path

huggingface-cli download --resume-download hfl/chinese-macbert-large
```

直接在命令行中使用也可，会提示使用该方法更快