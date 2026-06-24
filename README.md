# Huffman Coding · 文本与图像无损压缩

信息论课程设计：一套完整实现 **Huffman 编码** 的无损压缩系统，支持文本文件与图像。

## 功能

- **Huffman 引擎** —— 最优前缀码构造、规范码转换、基于树的解码
- **文本压缩** —— 字符级 Huffman 编码，通过码点表示完整支持 Unicode
- **图像压缩** —— 灰度与彩色图像的像素级编码；可选 **DPCM**（差分脉冲编码调制）+ MED 预测器，
  在平滑区域显著提升压缩率
- **信息论指标分析** —— 计算信源熵、平均码长、编码效率，以及与定长编码相比的压缩比
- **自动报告生成** —— 输出 Word 课程报告
- **可扩展架构** —— 设计上支持继续加入 LZW、算术编码等新压缩模块

## 项目结构

```
.
├── main.py                  CLI 入口
├── src/
│   ├── huffman.py           核心 Huffman 引擎
│   ├── encoder.py           文本编码
│   ├── decoder.py           文本解码
│   ├── image_codec.py       图像编解码（含 DPCM）
│   └── utils.py             位流 I/O、指标计算、文件工具
├── tests/                   单元测试（引擎 / 编解码 / 位流）
├── experiments/             文本与图像实验、对比实验、图表生成
├── data/                    测试语料（中英文文本、灰度图）与样本
├── results/tables/          实验数据 CSV
├── report/                  LaTeX 报告与参考文献
└── legacy/                  早期单文件版本（演进记录）
```

## 运行

```bash
pip install -r requirements.txt
python main.py --help
```

跑测试：

```bash
pytest tests/
```

跑实验并生成图表：

```bash
python experiments/run_text_experiments.py
python experiments/run_image_experiments.py
```

## 实验结果

实验数据在 [`results/tables/`](./results/tables/)：

| 文件 | 内容 |
| :--- | :--- |
| `text_experiments.csv` | 中英文语料的熵、平均码长、编码效率、压缩比 |
| `image_experiments.csv` | 灰度/彩色图像在有无 DPCM 下的压缩表现 |

核心观察：Huffman 编码的平均码长逼近信源熵，编码效率通常在 95% 以上；
对图像这类空间相关性强的数据，先做 DPCM 预测再编码能明显降低熵。

## 许可

[MIT](./LICENSE) © 2025 liuzunn
