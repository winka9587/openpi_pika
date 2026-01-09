# 介绍

pika数据的转换为lerobot格式数据集的转换工具

在openpi代码基础上开发，增加了目录: 

~~~
openpi/examples/pika
~~~

## 需要

在启动前你需要完成这些:

1. pick gripper采集数据, 并按照官方文档生成每个episode的hdf5文件
2. 配置openpi环境并激活

## 主要功能:
pika_cut_gradio.py: 基于gradio的web界面，对pika采集的序列数据进行裁剪和打标签。(注意: 目前的prompt默认为'pick the box'，user_name/repo默认为winka9587/pick_cillion_v3, 默认会在~/.cache/huggingface/lerobot/路径下创建~/.cache/huggingface/lerobot/winka9587/pick_cillion_v3)

convert_pika_data_to_lerobot.py: 已废弃，未经测试，但依然可以参考其中的加载hdf5逻辑
load_lerobot_test.py: 用于检查pika_cut_gradio.py转换后生成的数据(默认路径为~/.cache/huggingface/lerobot/，可以修改为你自己生成的测试)

# 可能遇到的问题

默认使用端口7850, 你可以修改为任意其他端口

如果异常退出导致端口持续占用可使用以下命令:

~~~
# 检查占用
lsof -ti:7850

# 关闭占用进程
lsof -ti:7850|xargs kill -9
~~~

# pika采集数据格式

生成的hdf5格式数据

~~~
"""
'camera', 
    'color', 
        'pikaDepthCamera_l',    (n, ), [bytes, bytes, ...]
        'pikaDepthCamera_r',    (n, ), [bytes, bytes, ...]
        'pikaFisheyeCamera_l',  (n, ), [bytes, bytes, ...]
        'pikaFisheyeCamera_r'   (n, ), [bytes, bytes, ...]
    'colorExtrinsic', 
        'pikaDepthCamera_l',    (4, 4)
        'pikaDepthCamera_r',    (4, 4)
        'pikaFisheyeCamera_l',  (4, 4)
        'pikaFisheyeCamera_r'   (4, 4)
    'colorIntrinsic', 
        'pikaDepthCamera_l',    (3, 3)
        'pikaDepthCamera_r',    (3, 3)
        'pikaFisheyeCamera_l',  (3, 3)
        'pikaFisheyeCamera_r'   (3, 3)
    'depth', 
        'pikaDepthCamera_l',    (n, ), [bytes, bytes, ...]
        'pikaDepthCamera_r'     (n, ), [bytes, bytes, ...]
    'depthExtrinsic', 
        'pikaDepthCamera_l',    (4, 4)
        'pikaDepthCamera_r'     (4, 4)
    'depthIntrinsic'
        'pikaDepthCamera_l',    (3, 3)
        'pikaDepthCamera_r'     (3, 3)
'gripper', 
    'encoderAngle', 
        'pika_l',               (n, ), [bytes, bytes, ...]
        'pika_r'                (n, ), [bytes, bytes, ...]
    'encoderDistance'
        'pika_l',               (n, ), [bytes, bytes, ...]
        'pika_r'                (n, ), [bytes, bytes, ...]
'instruction', 
'localization', 
    'pose'
        'pika_l'                (n, 6), [bytes, bytes, ...]
        'pika_r'                (n, 6), [bytes, bytes, ...]
'size', 
'timestamp'                     (n, ), [bytes, bytes, ...]
"""
~~~