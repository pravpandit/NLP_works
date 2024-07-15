# Chapter 9 The Harm of Large Models (Part 1)

## 9.1 Introduction

In this section, we will begin to explore the harms (harms) of large language models. In this course, we will cover several of these harms:

- Performance differences (this section)
- Social biases and stereotypes (this section)
- Harmful information (next section)
- False information (next section)

In addition, in subsequent courses, we will talk about other more aspects of harm:

- Security and privacy risks (future content)
- Copyright and legal protection (future content)
- Environmental impact (future content)
- Concentration of power (future content)

**Hazards of emerging technologies**: We know that "with great power comes great responsibility. For the current groundbreaking big models, we need to understand the close relationship between the capabilities and hazards of these models. The potential demonstrated by the capabilities of big models will lead to the widespread adoption of these models, but at the same time cause their harm.

Since the development of AI is a product of recent developments, the study and prevention of hazards is still a very new thing. Therefore, looking back at history, understanding the defense of hazards, safety and ethical issues in other fields in the past history, and first considering some high-level ideas and methods used in disciplines with mature hazard and safety traditions, can help to learn from the current AI field

**Belmont Report and IRB.**

- The Belmont Report was written in 1979 and outlined three principles (respect for people, benevolence andimpartiality).
- This report is the basis for Institutional Review Boards (IRBs). 
- IRBs are committees that review and approve research involving humans, acting as a proactive mechanism to ensure safety.

**Bioethics and CRISPR. **

- When the gene editing technology CRISPR CAS was created, the biomedical community established community standards prohibiting the use of these technologies for many forms of human gene editing.
- When community members are found to violate these standards, they are expelled from the community, reflecting the strict enforcement of community norms.

**FDA and Food Safety. **

- The Food and Drug Administration (FDA) is a regulatory agency responsible for setting safety standards.
- The FDA often conducts multiple stages of testing on foods and drugs to verify their safety.
- The FDA uses established theories from scientific disciplines to determine what to test.

In this course, we will focus on relatively specific but low-level concerns related to the hazards of LLMs. The current content focuses on the following two points:

**Harms related to performance differences**: As we saw in the section on the capabilities of large-scale language models, large language models can be adapted to perform specific tasks. For certain tasks (e.g., question answering), performance differences mean that the model performs better on some groups and worse on others. For example, an automatic speech recognition (ASR) system performs worse on black speakers than on white speakers.([Koenecke et al., 2020](https://www.pnas.org/content/117/14/7684)). Feedback loops (one in which large models continue to train as data accumulates) can amplify differences over time: if systems don’t work well for some users, they won’t use them and will generate less data, leading to even greater differences in future systems.

**Social bias and stereotype-related harms**: Social bias is the systematic association of a concept (e.g., science) with certain groups (e.g., men) relative to others (e.g., women). Stereotypes are a specific and pervasive form of social bias, where associations are widely held, oversimplified, and generally fixed. For humans, these associations come from rapid cognitive heuristics. They are particularly important for language technologies, where stereotypes are constructed, acquired, and propagated through language. Social bias can lead to performance differences, and large language models may perform poorly on data that demonstrate counter-stereotype associations if they cannot understand them.

## 9.2 Social Groups

In the United States, protected attributes are demographic characteristics that cannot be used as a basis for decision making, such as race, gender, sexual orientation, religion, age, nationality, disability, physical appearance, socioeconomic status, etc. Many of these attributes are often controversial, such as race and gender. These artificially constructed categories are different from natural divisions.Differently, existing work in AI often fails to reflect modern treatments of these attributes in the social sciences, for example, gender is not a simple binary division but a more fluid concept, as shown in research by Cao and Daumé III (2020) and Dev et al. (2021). 

While protected groups are not the only ones to focus on, they are a good starting point: the relevant groups vary across cultures and contexts (Sambasivan et al., 2021). In addition, we need to pay special attention to historically marginalized groups. In general, AI systems do not harm equally: groups that have historically been disempowered and discriminated against deserve special attention ([Kalluri, 2020](https://www.nature.com/articles/d41586-020-02003-2)). It is worth noting that it would be extremely unfair if AI systems further oppressed these groups. Poor performance of large language modelsHeterosocial bias often coincides with historical discrimination. Intersectionality theory (Crenshaw (1989)) suggests that individuals who are at the intersection of multiple marginalized groups (such as black women) often experience additional discrimination.

## 9.3 Quantifying performance differences/harms of social bias in LLMs

Large models are trained using large pre-training data, so data bias may lead to performance and social bias harms of large language models. Here we measure it through two examples.

**Name bias**

Here we first train the large model on SQuAD data, and then design a new task for testing.

- Motivation: Test the model's understanding and behavior in text involving names.
- Original task: [SQuAD - Stanford Question Answering Datasets (Rajpurkar et al., 2016) 
- Modified task: construct additional test examples using SQuAD data, swapping two names from previous test answers. Finally test the model's answer accuracy.
- Metric: Flips represent the percentage of name pairs that swapping names changes the model's output.

Results:

- Models generally predict names related to their famous person, consistent with their domain of expertise.
- The effect decreases quickly for less famous people.
- Models generally do not change their predictions when swapping names.

| Model | Parameters | Original acc. | Modified acc. | Flips |
| -------------------- | ---------- | ------------- | ------------- | ----- |
| RoBERTa-base | 123M | 91.2 |49.6 | 15.7 |
| RoBERTa-large | 354M | 94.4 | 82.2 | 9.8 |
| RoBERTA-large w/RACE | 354M | 94.4 | 87.9 | 7.7 |

Detailed results can be found in the [original paper](https://aclanthology.org/2020.emnlp-main.556.pdf).

**Stereotype**

- Motivation: Evaluate how models behave in text involving stereotypes 
- Task: Compare the model's probability for sentences with stereotype and counter-stereotype associations 
- Metric: The stereotype score is the proportion of stereotype examples that the model prefers. The authors say a score of 0.5 is ideal. 

Results:

- All models show a systematic preference for stereotyped data.
- Larger models tend to have higher stereotype scores.

| Model | Parameters | Stereotype Score |
| ------------ | ---------- | ---------------- |
| GPT-2 Small | 117M | 56.4 |
| GPT-2 Medium | 345M | 58.2 |
| GPT-2 Large | 774M | 60.0 |

## 9.4 Measurement and Decision Making

There are many fairness metrics that can be used to translate performance differences into a single measurement. However, many of these fairness metrics cannot be minimized simultaneously ([Kleinberg et al., 2016](https://arxiv.org/pdf/1609.05807.pdf)) and fail to meet the expectations of stakeholders for the algorithm ([Saha et al., 2020](https://arxiv.org/pdf/2001.00089.pdf)).
Many design decisions for measuring bias can significantly change the results, such as vocabulary, decoding parameters, etc. ([Antoniak and Mimno, 2021](https://aclanthology.org/2021.acl-long.148.pdf)). Existing benchmarks for large language models (LLMs) have been heavily criticized ([Blodgett et al.Many upstream bias measures do not reliably predict downstream performance differences and substantive harms (Goldfarb-Tarrant et al., 2021). 

## 9.5 Other considerations

LLMs have the potential to cause harms in a variety of ways, including performance differences and social biases. Understanding the social impact of these harms requires considering the social groups involved and their conditions, such as historical marginalization and lack of power. While harms are often better understood in the context of specific downstream applications, LLMs are the foundational model for upstream.

## 9.6 Decision-making issues

Existing approaches often fail to effectively reduce or address these harms; in practice, many technical mitigations work poorly. Sociotechnical approaches that encompass the broader ecosystem in which LLMs are situated may be necessary to significantly mitigate these harms.

## Further reading
- [Bommasani et al., 2021](https://arxiv.org/pdf/2108.07258.pdf)
- [Bender and Gebru et al., 2021](https://arxiv.org/pdf/2108.07258.pdf)l., 2020](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)
- [Blodgett et al., 2020](https://aclanthology.org/2020.acl-main.485.pdf)
- [Blodgett et al., 2021](https://aclanthology.org/2021.acl-long.81.pdf)
- [Weidinger et al., 2021](https://arxiv.org/pdf/2112.04359.pdf)