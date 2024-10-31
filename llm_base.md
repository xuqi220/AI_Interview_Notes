1. **Prefix LM 和 Causal LM 区别是什么？**
    
    Prefix LM（前缀语言模型）和Causal LM（因果语言模型）是两种不同类型的语言模型，它们的区别在于生成文本的方式和训练目标。
    
    **Prefix LM**可以看作是Encoder-Decoder模型的变体，前缀中的任意Token间是互相可见的，采用Auto Encoding (AE-自编码)模式。待生成的Token采用的是Auto Regressive (AR-自回归)模式，待生成的Token只能看到前缀中的Tokens。具体例子可以参考ChatGLM系列。
 
    **Causal LM** 是因果语言模型，采用了Transfomer的Decoder，采用Auto Regressive模式，就是根据t时刻之前的token来预测下t时刻的token，主要通过attention_mask实现，目前流行地大多数模型都是这种结构。代表模型有GPT系列、Qwen系列、LLama系列。

2. **大模型LLM的训练目标**

    语言模型通常用来计算文本序列的概率，大语言模型通过大量的语料，采用极大似然估计（Maximum Likelihood Estimation，MLE）估计模型参数。具体来说，大语言模型的训练任务是给定t时刻之前的tokens预测t时刻token，并最大化在语料库中出现的文本序列的概率。

3. **大模型架构**
    Transformer 模型一开始是用来做 seq2seq 任务的，所以它包含 Encoder 和 Decoder 两个部分；他们两者的区别主要是，Encoder 在抽取序列中某一个词的特征时能够看到整个序列中所有的信息，即上文和下文同时看到；而 Decoder 中因为有 mask 机制的存在，使得它在编码某一个词的特征时只能看到自身和它之前的文本信息。

    几种主要的架构:

    以BERT为代表的encoder-only
    以T5和BART为代表的encoder-decoder
    以GPT为代表的decoder-only，
    以UNILM、ChatGLM为代表的PrefixLM(前缀部分是双向，后面将要生成的部分是单向的)

4. **涌现能力是什么？原因？**

    涌现能力（Emergent Ability）是指模型在训练过程中能够生成出令人惊喜、创造性和新颖的内容或行为。这种能力使得模型能够超出其训练数据所提供的内容，并产生出具有创造性和独特性的输出。

5. **为何现在的大模型大部分是Decoder only结构**
* **效率**：decoder-only支持一直复用KV-Cache，对多轮对话更友好，每个Token的表示之和它之前的输入有关，有利于token表示复用，而encoder-decoder和PrefixLM，需要考虑新生成的token，因此难以做到。
* **经验**：就生成任务而言，引入双向注意力并无实质好处。有可能损害模型的创造性。

6. **什么是LLMs复读机问题？什么原因造成的？如何缓解？**

7. **LLMs输入句子长度理论上可以无限长吗？**

8. **大模型的适用场景**

9. **如何让大模型处理更长的文本？**

