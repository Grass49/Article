# 意见6：ONNX与OM真实图片配对验证

## 审稿人原文
Comment 6: Sections 3.4.1 and 4.1.2 (numerical consistency)

The use of cosine similarity on 10 random inputs is a useful initial check, but it provides limited evidence that model behavior is preserved, especially under FP16 settings. Evaluate a representative held-out sample and report task-level agreement for the classifiers together with one output-error measure. Clarify how outputs were compared across batch sizes and how the 0.999 threshold was selected. Cosine similarity can remain as the admission criterion, but its limitations should be acknowledged.

中文：10个随机输入的余弦检查不足以证明部署后行为保持，尤其是FP16。用具有代表性的留出样本，报告分类任务的一致性和一个输出误差；说明跨batch输出如何对应、0.999为何这样设定；余弦可以保留为准入条件，但需说明局限。

## 这份工具做什么
- prepare：电脑运行ONNX，抽取固定图片，生成参考输出和OM所需输入bin。
- run-om：昇腾设备使用现有msame运行指定OM。
- compare：逐图配对，计算Top-1一致率、余弦相似度和相对L2误差。
- 先验证Exp67（论文记录batch=4）。这不是重新运行搜索，不增加TOPSIS指标，也不新增分类一致率准入阈值。
- 当前未知模型的训练来源，所以工具写为外部测试样本，不保证未参与原模型训练。不因数据集叫ImageNetV2就自动确认held-out。
- 100张是小规模检查方案，不是审稿人规定的最低数量。只验证Exp67不能推广到81组。
- 预处理采用标准TorchVision ImageNet约定：RGB、短边256（双线性）、中心裁剪224、除255、mean=[.485,.456,.406]、std=[.229,.224,.225]、NCHW。这不等于已经确认下载权重的原始预处理。必须确保ONNX/OM内部没有重复的预处理或不同类别顺序。
- 工具不读取或执行.pth。使用你论文中实际导出的同一份ONNX和从它编译的OM。

## 你需要准备
1. 已解压的ImageNetV2 MatchedFrequency图片目录，直接指向包含类别子目录的根目录。
2. 电脑端ONNX：已知为 E:/workspace/onnx/resnet18_full.onnx。
3. 昇腾设备上的Exp67 OM实际路径。
4. 设备上已经可正常执行的msame路径及CANN运行环境。
5. 确认OM输入为静态[4,3,224,224]、输入接口float32、单个1000类输出。如果不是，按真实接口修改参数；不要把内部force_fp16误当成输入接口也是float16。

## 第一步：电脑安装依赖并生成输入
在本代码文件夹打开终端：
~~~text
python -m pip install -r requirements_pc.txt
~~~
将下面的图片路径替换为你的实际目录。其余ONNX路径来自本次已找到的文件。
~~~text
python validate_onnx_om.py prepare --images "E:/datasets/imagenetv2-matched-frequency-format-val" --onnx "E:/workspace/onnx/resnet18_full.onnx" --job "E:/workspace/comment6_exp67" --samples 100 --seed 42 --batch-size 4
~~~
路径仅是运行示例，代码不会下载数据集。
默认按文件夹平衡抽样：随机打乱类别文件夹，各文件夹轮流抽图片；如果有至少100个类别目录，100张会覆盖100个目录。文件夹不自动转换成真实类别标签。可用 --sampling random 改为全体随机抽样。
抽样独立于模型预测结果，使用固定种子。输入图片内容重复时跳过重复字节文件；不会根据预测正确与否挑选。
图片解码失败会报错，不悄悄跳过失败样本。
job目录必须是新目录；防止覆盖已有实验。

电脑上没有昇腾NPU也能完成这一步。
ONNX如果固定batch=1，参考模型按batch=1逐张执行；动态batch默认使用4。OM仍按实际batch=4执行。两边使用相同逐图预处理值，最终按图片清单配对。
如果ONNX固定其他batch，脚本按其真实batch运行并填充最后一批；不修改ONNX图来绕过输入限制。

## 第二步：复制到昇腾设备并运行OM
将以下内容复制到昇腾设备：
- validate_onnx_om.py
- 第一步生成的整个comment6_exp67文件夹（包含reference_outputs.npy、manifest.json、samples.json、inputs等）

设备仅需numpy；不必安装ONNX Runtime或PyTorch。
~~~text
python3 -m pip install numpy
~~~
保持你已有的CANN环境；先用现有方式确认msame能正常运行。
下面所有 /path/to/... 都换成设备实际路径：
~~~text
python3 validate_onnx_om.py run-om --job /path/to/comment6_exp67 --om /path/to/model_exp67.om --msame /path/to/msame --run /path/to/exp67_real_images_run1 --config-id Exp67
~~~
默认device=0。如需其他设备，增加 --device 1。
默认静态OM。只有编译时实际使用动态batch的OM，才加 --dynamic-batch。
默认选第0个OM输出；多输出图需用 --output-index 选择与ONNX --output-name对应的分类输出。
--output-dtype auto 根据已知batch×1000对应的字节数区分FP16/FP32。也可明确指定 --output-dtype float32 或 float16。必须先确保OM真实batch与prepare参数一致，字节数不能独立证明模型语义。
每批独立msame调用，--loop 1，输入是该批的真实图片bin。每批日志和原始输出单独保存。最后一批不足时重复最后一张填充，但统计时只取真实样本行。
因为每批重新加载模型，这段代码不能用于论文的延迟/吞吐量计时实验。
不能复用旧run目录。运行失败就报错并保存failed状态，不用缺失值或旧输出继续算成功率。

## 第三步：生成对比结果
可以在设备运行，也可以把完整run目录复制回电脑后运行。
~~~text
python3 validate_onnx_om.py compare --job /path/to/comment6_exp67 --run /path/to/exp67_real_images_run1
~~~
电脑示例：
~~~text
python validate_onnx_om.py compare --job "E:/workspace/comment6_exp67" --run "E:/workspace/exp67_real_images_run1"
~~~
compare重复运行时应使用新的报告目录，例如 --report /path/to/report_v2，避免覆盖第一次结果。

## 应把哪些结果发回来
run/report下：
- summary.json：图片数、分类一致率、相对L2的均值/中位数/P95/最大值、余弦分布、0.999通过数。
- per_image.csv：每张图片的ID、ONNX/OM预测类别、是否一致、余弦值和相对L2误差。
- results.md：简短可读结果。
- provenance.json：模型校验值、batch口径、预处理、样本和运行信息。
另保留job/samples.json和设备各批msame.log，便于追溯。
报告将逐图Top-1一致率与真实标签准确率明确区分；本工具不计算后者，因为你的类别映射和权重来源尚未确认。

## 指标与判断
Top-1 agreement = argmax(ONNX输出)与argmax(OM输出)相同的真实图片数 / N。
Relative L2 = ||OM - ONNX||2 / max(||ONNX||2, epsilon)，默认epsilon=1e-12。
误差直接在所选输出张量上计算，默认logits。不为了让误差变小而额外加softmax或对输出逐图缩放。
如果模型本来输出概率，prepare加 --output-kind probabilities；compare会检查非负与行和。ONNX和OM须处在相同输出位置。
余弦对零范数向量未定义：记录为空并计入undefined，绝不当作通过。其他NaN/Inf直接报错。
0.999沿用原论文筛查值，只记录本次真实图片上的通过情况；工具不根据结果调阈值，也不替作者编造依据。可在prepare用 --threshold-rationale "实际依据"记录你真实的说明。
即使余弦接近1，也可能出现分类改变或较大幅度误差，所有差异均保留。

## 测试其他OM / batch时
同batch的其他OM可复用同一个job，换OM路径、config-id和新的run目录。
不同batch必须新建job。例如batch=16，沿用原100张图片：
~~~text
python validate_onnx_om.py prepare --images "E:/datasets/imagenetv2-matched-frequency-format-val" --onnx "E:/workspace/onnx/resnet18_full.onnx" --job "E:/workspace/comment6_batch16" --sample-list "E:/workspace/comment6_exp67/samples.json" --batch-size 16
~~~
指定sample-list时，以清单全部样本为准，--samples不重新抽样。它校验图片SHA-256，保证换batch没有偷偷换图。
100张、batch16会有7批，最后12个填充位置不计入样本数，最终仍是100张。
这只验证所选OM与ONNX的部署一致性，不是所有设备/配置的普遍保证。

## 软件验证与限制
运行软件单元测试：
~~~text
python -m unittest -v test_validation
~~~
这些测试故意使用生成图片和模拟推理来检查程序的配对、填充、度量和错误拦截；不是模型实验数据，不写入论文，不计入研究结果。
当前本机没有ONNX Runtime或Ascend NPU运行环境，未完成真实ONNX/OM端到端实验。真实运行需要你按上述步骤在相应环境执行。
仅支持单输入NCHW图像分类图及已知固定大小分类输出；不直接支持检测器、动态分辨率、多输入模型或AIPP输入。

## 核对依据
- Ascend msame： https://gitee.com/ascend/tools/blob/master/msame/README.md
- ONNX Runtime Python API： https://onnxruntime.ai/docs/api/python/api_summary
- TorchVision ResNet-18预处理： https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- ImageNetV2作者说明： https://github.com/modestyachts/ImageNetV2
