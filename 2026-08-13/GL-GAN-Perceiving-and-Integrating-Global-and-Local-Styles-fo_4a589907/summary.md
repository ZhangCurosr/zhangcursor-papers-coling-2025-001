---
title: "GL-GAN-Perceiving-and-Integrating-Global-and-Local-Styles-fo"
source: https://aclanthology.org/2025.coling-main.166.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:33:03"
---

# 论文速读：GL-GAN: Perceiving and Integrating Global and Local Styles for Handwritten Text Generation with Mamba

## 一句话总结
本文提出GL-GAN，一种结合卷积与视觉状态空间模型（VSS/Mamba）的离线手写文本生成框架，通过混合风格编码器（HSE）与动态特征增强模块（DFEM）显式感知并融合全局/局部多层笔迹风格，在IAM与CVL基准上显著超越现有SOTA方法。

## 研究问题与动机
- 现有离线手写文本生成（HTG）模型（如VATr/VATr++、HWT）缺乏对笔迹风格的有效感知与整合能力，导致合成样本的真实感与风格保真度受限。
- 单一风格编码器难以同时精准捕捉全局（长程布局、整体趋势）与局部（笔画纹理、墨迹细节）特征，造成风格表征不完整。
- 现有方法普遍忽视不同层级风格特征间的纠缠关系，直接拼接或逐元素相加会引入冗余信息并削弱风格多样性。
- 受神经科学分层视觉处理机制启发，作者认为HTG的核心瓶颈在于模型无法像人脑一样分层次感知多维笔迹风格并进行认知级融合。

## 核心贡献（创新点）
- 提出GL-GAN生成框架，首次将视觉状态空间模型（VSS/Mamba）引入HTG任务，以并行双分支结构同时捕获全局与局部笔迹风格。
- 设计混合风格编码器（HSE），结合残差块（CNN）与VSS块（2D-SSM），利用不同感受野互补提取多层级风格特征，突破单一骨干网的感知边界。
- 提出动态特征增强模块（DFEM），内嵌ASEM与DSEM，自适应建模多层风格间的纠缠关系并剔除冗余细节，替代传统的简单特征相加操作。
- 构建对抗损失、文本识别CTC、写作者分类CE与循环一致性联合优化框架，在IAM（IND）与CVL（OOD）数据集上取得全面SOTA性能，并验证了生成样本对下游HTR任务的显著增益。

## 方法详解
- **整体架构**：采用条件GAN范式，生成器$G$分为编码器$G_\varepsilon$与解码器$G_d$，判别器$D$、文本识别器$R$、写作者分类器$W$联合训练，总损失$L = L_D + L_R + L_W + L_C$。
- **混合风格编码器（HSE）**：输入风格样本$X_w$后并行分流：残差块分支提取局部风格特征$F_l$；VSS块分支基于2D-Selective-Scan（CSM四向扫描+S6状态空间模块+Scan Merge）提取全局风格特征$F_g$。VSS backbone堆叠4个Block，通道数匹配为[64,128,256,512]。
- **动态特征增强模块（DFEM）**：接收$F_g$与$F_l$，先经ASEM通过注意力机制挖掘互补细节与纹理，再经DSEM从通道与空间双维度挤压增强以抑制冗余，逐元素相加得到纠缠特征序列$F_{gl}$。
- **Transformer编码与解码**：$F_{gl}$经3层Transformer编码器（8头自注意力）聚合为多层级风格特征$Z_s$；文本$C$经线性层得$Q_t$，过自注意力得$Q_{re}$，再通过多头交叉注意力（MHCA）将$Z_s$与$Q_{re}$融合为风格-文本纠缠序列$F_{st}$，最后经4层残差上采样模块输出合成图像$Y_w^C$。
- **损失函数设计**：$L_D$为WGAN风格对抗损失；$L_R$为CTC文本识别损失（强制内容准确）；$L_W$为写作者分类交叉熵损失（强制风格身份一致）；$L_C=\mathbb{E}[\|G_\varepsilon(X_w)-G_\varepsilon(Y_w^C)\|_1]$为循环一致性损失，约束生成样本在编码器空间中保持原风格分布。

## 实验与结果
- **数据集**：IAM（500作者，训练339/测试161）与CVL（311作者，抽取27作者用于OOD跨域泛化测试）。
- **评估基线**：GANwriting、HWT、VATr、VATr++（公平对比采用官方权重或代码复现）。
- **主要定量结果（IAM IND）**：GL-GAN在FID(14.32)、HWD(0.3498)、KID(0.595)三项指标上均最优；相对最强基线VATr++分别提升12.09%、11.26%、15.12%。
- **主要定量结果（CVL OOD）**：GL-GAN在FID(26.03)、HWD(1.1357)、KID(1.551)上
