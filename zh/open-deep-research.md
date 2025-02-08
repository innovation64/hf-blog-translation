---
title: "开源深度研究-解放我们工作的搜索智能体"
thumbnail: /blog/assets/open-deep-research/thumbnail.png
authors:
- user: m-ric
- user: albertvillanova
- user: merve
- user: thomwolf
- user: clefourrier
translators:
- user: innovation64
---

# 开源深度研究-解放我们工作的搜索智能体

## 简要概述

昨天，OpenAI 发布了 Deep Research(深度研究)，一款可以浏览网页以总结内容并基于总结回答问题的系统。该系统令人印象深刻，我们第一次尝试时简直震撼了。

博客中的一个主要成果是，在 General AI Assistants Benchmark (GAIA) 测评上显著提高了表现——这是我们最近也在尝试的一个基准测试。OpenAI 成功实现了 1-shot (1 个样例)问题的正确回答率接近 67%，以及在特别具有挑战性的 “Level 3” (第三等级) 问题（需要多步骤推理和工具使用）上有 47.6% 正确率（下文有 GAIA 的介绍）。

Deep Research 由一个大型语言模型（LLM）构成（可以从 OpenAI 提供的 LLM 列表中选择，比如 4o、o1、o3 等），以及一个内部的“智能体框架”，它指导 LLM 使用像网页搜索这样的工具，并将其行动组织为多个步骤。

虽然现在强大的 LLM 已经可以自由获取（例如，最近发布的 DeepSeek R1 模型），OpenAI 并未透露 Deep Research 背后智能体框架的具体细节……

于是，我们决定展开一场 24 小时的任务，复现他们的结果，并在此过程中将所需的框架开源！

时间在倒计时，出发！⏱️

目录
	•	什么是智能体框架，为什么它们很重要？
	•	GAIA基准测试
	•	构建开源Deep Research
	•	使用CodeAgent
	•	选择合适的工具 🛠️
	•	结果 🏅
	•	社区复现
	•	最重要的下一步

## 目录

- [什么是智能体框架，为什么它们很重要?](#什么是智能体框架，为什么它们很重要)
- [GAIA 基准测试](#GAIA-基准测试)
- [构建开源 Deep Research](#构建开源-Deep-Research)
  - [使用CodeAgent](#使用-codeagent)
  - [选择合适的工具 🛠️](#选择合适的工具-🛠️)
- [结果 🏅](#结果-🏅)
- [社区复现](#社区复现)
- [最重要的下一步](#最重要的下一步)


## 什么是智能体框架，为什么它们很重要?

> [!TIP]
>   智能体框架是位于大型语言模型（LLM）之上的一层，使得该 LLM 执行特定的操作（比如浏览网页或读取 PDF 文档），并将其操作组织成一系列步骤。
>	如果你想快速了解智能体的概念，可以查看[吴恩达](https://youtu.be/sal78ACtGTc?feature=shared&t=52) 的精彩访谈 和我们关于 smolagents 库的介绍[博客](https://huggingface.co/blog/smolagents)。如果你想更深入了解智能体，可以订阅我们即将开课的智能体课程：[点击此处](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=9ed45a3ef6)。   

几乎每个人都已经体验过大型语言模型（LLMs）在与聊天机器人互动时的强大能力。然而，并不是每个人都意识到，将这些 LLM 集成到智能体系统中，可以赋予它们真正的超级能力！

这是一个最近的例子，比较了几种前沿 LLM 在有无智能体框架（在此案例中为简单的 [smolagents](https://github.com/huggingface/smolagents) 库）下的表现——使用智能体框架可以将性能提升多达 60 个百分点！

![Benchmarks](https://huggingface.co/datasets/huggingface/documentation-images/resolve/6c7ed2035810565043c92b472d5564c3f1fa4d7e/blog/open-deep-research/benchmarks.png)

事实上，OpenAI 也在[发布博客](https://openai.com/index/introducing-deep-research/) 中强调，Deep Research 在知识密集型的 “[Humanity’s Last Exam](https://huggingface.co/datasets/cais/hle)” 基准测试中的表现，远远优于独立的 LLM。

那么，当我们将目前顶尖的 LLM 集成到智能体框架中，朝着 `open-DeepResearch` 目标前进时，会发生什么呢？

**简短说明**： 我们将基于相同的 GAIA 挑战对我们的结果进行基准测试，但请记住，这仍然是一个进行中的工作。DeepResearch 是一项巨大的成就，其开源复现将需要时间。特别是，完全一致性将需要改进的浏览器使用和交互功能，例如 OpenAI Operator 提供的那样，也就是说，超越当前我们在第一步中探索的仅文本网页交互。

让我们首先了解挑战的范围：GAIA。


## GAIA 基准测试


[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) 被认为是目前最全面的智能体基准测试。它的问题非常困难，涵盖了许多基于 LLM 的系统面临的挑战。以下是一个难题示例：

> 在 2008 年的画作《乌兹别克斯坦的刺绣》中显示的水果，哪些是 1949 年 10 月的海洋客轮早餐菜单的一部分？该客轮后来被用作电影《最后的航程》的浮动道具。请按顺时针顺序列出这些水果，起始位置从 12 点钟方向开始，水果名称请使用复数形式。

你可以看到，这个问题涉及了几个挑战：

- 以受限格式回答，
- 使用多模态能力（从图像中提取水果），
- 收集多个信息片段，其中有些信息依赖于其他信息：
    - 识别图片中的水果
    - 查找哪艘海洋客轮被用作《最后的航程》的浮动道具
    - 查找该海洋客轮的 1949 年 10 月早餐菜单
- 将解决问题的步骤按正确顺序排列。

解决这个问题需要高水平的规划能力和严格的执行力，这正是 LLM 在单独使用时容易遇到困难的两个领域。

因此，这是一个非常适合智能体系统的优秀测试集！

在 GAIA 的 [公开排行榜](https://huggingface.co/spaces/gaia-benchmark/leaderboard) 上，当 GPT-4 没有使用任何智能体设置时，它在验证集上的得分甚至没有达到 7%。而在 Deep Research 的支持下，OpenAI 在验证集上的得分达到了 67.36%，提升了一个数量级！(尽管我们不知道它们在私有测试集上的表现如何。)

让我们看看能否通过开源工具做得更好！

[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) is arguably the most comprehensive benchmark for agents. Its questions are very difficult and hit on many challenges of LLM-based systems. Here is an example of a hard question:


## 构建开源 Deep Research

### 使用 CodeAgent

我们将要解决的第一个传统 AI 智能体系统的改进是使用所谓的“代码智能体”。正如 [Wang 等人 (2024)](https://huggingface.co/papers/2402.01030) 所展示的，让智能体以代码形式表达其动作有几个优势，最显著的优势是 **代码专门设计用来表达复杂的动作序列**。

考虑 Wang 等人给出的这个例子：

![Code Agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/6c7ed2035810565043c92b472d5564c3f1fa4d7e/blog/open-deep-research/code_agent.png)

这突出了使用代码的几个优势：

- 代码动作比 JSON **简洁得多**。
    - 需要执行 4 个并行流，每个流包含 5 个连续的动作？在 JSON 中，你需要生成 20 个 JSON 块，每个动作一个；而在代码中，这只需要 1 步。
    - 平均而言，论文显示代码动作比 JSON 少需要 30% 的步骤，这意味着生成的 token 数量相应减少。由于 LLM 调用通常是智能体系统的主要成本来源，这意味着你的智能体系统运行成本大约减少 30%。
- 代码可以重用来自常见库的工具
- 基准测试中表现更好，原因有二：
    - 更直观的表达动作方式
    - LLM 在训练中大量接触代码

以上优势已经通过我们在 [agent_reasoning_benchmark](https://github.com/aymeric-roucher/agent_reasoning_benchmark) 上的实验得到了验证。

在构建 `smolagents` 时，我们还可以提到一个显著的附加优势，那就是更好的状态管理：这对于多模态任务特别有用。需要存储这张图片/音频/其他内容以供后续使用？没问题，只需将其分配为状态中的变量，若需要，你可以在 4 步后再次使用它。在 JSON 中，你需要让 LLM 将其命名为字典键，并相信 LLM 以后会理解它仍然可以使用它。

### 选择合适的工具 🛠️

现在，我们需要为智能体提供一套合适的工具。

**1.** 一个网页浏览器。虽然像 [Operator](https://openai.com/index/introducing-operator/) 这样的完全功能的网页浏览器交互将是达到最佳性能所必需的，但我们目前开始使用一个极其简单的基于文本的网页浏览器作为第一个概念验证。你可以在 [这里](https://github.com/huggingface/smolagents/blob/gaia-submission-r1/examples/open_deep_research/scripts/text_web_browser.py) 找到代码。

**2.** 一个简单的文本检查器，用于能够 **读取各种文本文件格式**，可以在 [这里](https://github.com/huggingface/smolagents/blob/gaia-submission-r1/examples/open_deep_research/scripts/text_inspector_tool.py) 找到。

这些工具来源于微软研究院的优秀智能体 [Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)，感谢他们！我们没有对它们做太多改动，因为我们的目标是尽可能以最低的复杂度获得最高的性能。

以下是我们认为可以显著提升这些工具性能的改进路线图（欢迎提交 PR 贡献！）：

- 扩展可以读取的文件格式数量。
- 提供更精细的文件处理方式。
- 用基于视觉的网页浏览器替代当前的网页浏览器，我们已经开始在 [这里](https://github.com/huggingface/smolagents/blob/gaia-submission-r1/src/smolagents/vision_web_browser.py) 进行这一工作。

## 结果 🏅

在我们的 24 小时以上的复现冲刺中，我们已经看到智能体在 GAIA 上的性能稳定提升！

我们迅速突破了之前使用开源框架的 SoTA（大约 46% 来自 Magentic-One），达到了目前在 [验证集上的 55.15% 性能](https://huggingface.co/spaces/gaia-benchmark/leaderboard)。

这种性能的提升主要归功于让我们的智能体用代码来表达它们的动作！事实上，当切换回使用 JSON 而非代码来写动作的标准智能体时，同样的设置下，性能立即下降至验证集上的 33% 平均得分。

[这是最终的智能体系统。](https://github.com/huggingface/smolagents/tree/gaia-submission-r1/examples/open_deep_research)

我们为你设置了 [一个实时演示](https://m-ric-open-deep-research.hf.space)，欢迎尝试！


<iframe
	src="https://m-ric-open-deep-research.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

然而，这仅仅是开始，还有很多需要改进的地方！我们的开源工具可以做得更好，smolagents 框架也可以进一步优化，我们也很期待探索使用更强大的开源模型来支持智能体的性能。

我们欢迎社区加入我们这一事业，共同利用开源研究的力量，构建一个出色的开源智能体框架！这将使任何人都能够在家里运行类似 DeepResearch 的智能体，使用自己喜欢的模型，采用完全本地化和定制化的方法！

## 社区复现

在我们专注于 GAIA 的同时，社区中也出现了其他一些优秀的开源 Deep Research 实现，具体来自以下几位：

- [dzhng](https://x.com/dzhng/status/1886603396578484630)
- [assafelovic](https://github.com/assafelovic/gpt-researcher)
- [nickscamara](https://github.com/nickscamara/open-deep-research)
- [jina-ai](https://github.com/jina-ai/node-DeepResearch)
- [mshumer](https://x.com/mattshumer_/status/1886558939434664404)

这些实现使用了不同的库来索引数据、浏览网页和查询 LLM。在这个项目中，我们希望 **复现 OpenAI 提出的基准测试（pass@1 平均得分），并基于切换到开源 LLM（如 DeepSeek R1）、使用视觉模型、比较传统工具调用与代码本地智能体的性能，进行基准测试并记录我们的发现。**

## 最重要的下一步

OpenAI 的 Deep Research 很可能得益于他们引入的优秀网页浏览器 [Operator](https://openai.com/index/introducing-operator/)。

所以我们接下来要解决的就是这个问题！从更一般的角度来看：我们将构建 GUI 智能体，也就是“能够查看你的屏幕并直接与鼠标和键盘进行交互的智能体”。如果你对这个项目感到兴奋，并且希望通过开源帮助每个人获得这样的酷炫功能，我们非常欢迎你的贡献！

我们还在 [招聘全职工程师](https://apply.workable.com/huggingface/j/AF1D4E3FEB/)，帮助我们一起推进这个项目。如果你有兴趣，快来申请吧🙂

- 要开始使用 Open Deep Research，可以尝试 [这里](https://github.com/huggingface/smolagents/tree/gaia-submission-r1/examples/open_deep_research) 的示例。
- 查看 [smolagents](https://github.com/huggingface/smolagents) 仓库。
- 阅读更多关于 smolagents 的 [文档](https://huggingface.co/docs/smolagents/index)，以及 [介绍博客](https://huggingface.co/blog/smolagents)。
