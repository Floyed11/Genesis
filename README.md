# Genesis Project

## 模型的工作过程：

- 数据清理
- 模型运行
- 结果输出

### 数据清理

/genesis/genesis/spiders/spec.py：爬取基础数据并存储在genesis/genesis/spec_cfp2017.csv中。

/spider1.py：爬取原始数据并保存在./spider_spec_cfp2017.csv中。

/genesis/genesis/clean.py：清理数据，从./spec_cfp2017.csv获取，简单地将内存等选项格式化为数值的形式，获得./clean.csv。

/clean.py：清理数据，从./spider_spec_cfp2017.csv获取，将内存等选项格式化为数值的形式，获得./clean.csv。

/genesis/genesis/spiders/trans.py：输入./clean.csv，将表变为数值和布尔型的组合，方便后续处理。获得./output_x.csv和./output_y.csv。

### 模型运行

文件存储在./model中。
