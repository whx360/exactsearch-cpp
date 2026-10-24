# 参考、署名与许可

- Bentley, J. L. (1975). *Multidimensional Binary Search Trees Used for Associative Searching*. CACM 18(9), 509–517. [DOI](https://doi.org/10.1145/361002.361007)。借鉴 KD-tree 思想；本项目实现中位数分区、AABB 下界和确定性 Top-k。树查询没有任意数据上的对数复杂度保证。
- C++ 公开工作草案：[nth_element](https://eel.is/c++draft/alg.nth.element)、[sort](https://eel.is/c++draft/sort)、[vector](https://eel.is/c++draft/vector)、[shared_ptr](https://eel.is/c++draft/util.smartptr.shared)、[数据竞争](https://eel.is/c++draft/intro.races)、[steady_clock](https://eel.is/c++draft/time.clock.steady)。草案会更新，不替代目标 C++17 编译器的实际验证。
- Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*. [UCI / DOI](https://doi.org/10.24432/C50P49)，CC BY 4.0。使用 [scikit-learn 1.5.2 的 digits.csv.gz](https://github.com/scikit-learn/scikit-learn/blob/1.5.2/sklearn/datasets/data/digits.csv.gz) 固定子集，1797 行而非完整 5620 行。下载脚本保留来源、身份哈希与切分方法；原始数据不随代码分发。
- 历史 Windows 实验使用 [w64devkit 2.0.0](https://github.com/skeeto/w64devkit/releases/tag/v2.0.0) / GCC 14.2。工具链不随项目分发，遵循各自许可。

堆、KD-tree、分支限界和线程池均为已有思想。项目内编写了距离计算、有界堆、树构建/查询、文件解析、线程调度和测试，未调用检索库替代核心算法。C++ 标准库提供容器、sort/nth_element、智能指针和线程基础设施；Python/NumPy 提供数据生成与独立数值参考。

实现、检查和文档整理使用了 AI 辅助；这不构成算法原创或完全独立人工开发的声明。未来引入第三方代码片段时，应逐文件补充来源、版本、许可证、修改和保留通知。

项目代码采用 MIT。外部数据及编译器、标准库保持原许可；项目 MIT 不改变其使用条件。
