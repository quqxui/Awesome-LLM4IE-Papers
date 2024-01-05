
# Awesome-LLM4IE-Papers


Awesome papers about generative Information extraction using LLMs

<p align="center" width="80%">
<img src="./image/intro.png" style="width: 50%">
</p>

The organization of papers is discussed in our survey: [Large Language Models for Generative Information Extraction: A Survey](https://arxiv.org/abs/2312.17617). 

If you have any suggestions or come across any mistakes, missing information, please feel free to let us know via email at **derongxu@mail.ustc.edu.cn**. We appreciate your feedback and help in improving our work.

If you find our survey useful for your research, please cite the following paper:

    @misc{xu2023large,
        title={Large Language Models for Generative Information Extraction: A Survey}, 
        author={Derong Xu and Wei Chen and Wenjun Peng and Chao Zhang and Tong Xu and Xiangyu Zhao and Xian Wu and Yefeng Zheng and Enhong Chen},
        year={2023},
        eprint={2312.17617},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    
## Table of Contents
- [Information Extraction tasks](#information-extraction-tasks)
    - [Named Entity Recognition](#named-entity-recognition)
    - [Relation Extraction ](#relation-extraction)
    - [Event Extraction](#event-extraction)
    - [Universal Information Extraction](#universal-information-extraction)
- [Learning Paradigms](#learning-paradigms)
    - [Supervised Fine-tuning](#supervised-fine-tuning)
    - [Few-shot ](#few-shot)
    - [Zero-shot](#zero-shot)
    - [Data Augmentation](#data-augmentation)
- [Specific Domain](#specific-domain)
- [Evaluation and Analysis](#evaluation-and-analysis)
- [Datasets](#datasets)


# Information Extraction tasks
A taxonomy by various tasks.

## Named Entity Recognition 
Models targeting only ner tasks.

### Entity Typing
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Calibrated Seq2seq Models for Efficient and Generalizable Ultra-fine Entity Typing](https://aclanthology.org/2023.findings-emnlp.1040/)  |   EMNLP Findings      |  2023-12   | [Github](https://github.com/yanlinf/CASENT) |
|  [Generative Entity Typing with Curriculum Learning](https://arxiv.org/abs/2210.02914)  |   EMNLP       |  2022-12   | [Github](https://github.com/siyuyuan/GET) |

### Entity Identification & Typing 
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [2INER: Instructive and In-Context Learning on Few-Shot Named Entity Recognition](https://aclanthology.org/2023.findings-emnlp.259/)  |   EMNLP Findings    |  2023-12   |    |
|  [In-context Learning for Few-shot Multimodal Named Entity Recognition](https://aclanthology.org/2023.findings-emnlp.196/)  |   EMNLP Findings    |  2023-12   |    |
|  [Large Language Model Is Not a Good Few-shot Information Extractor,<br> but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)  |   EMNLP Findings    |  2023-12   |  [Github](https://github.com/mayubo2333/LLM-IE)  |
|  [Learning to Rank Context for Named Entity Recognition Using a Synthetic Dataset](https://arxiv.org/abs/2310.10118)  |   EMNLP     |  2023-12   |  [Github](https://github.com/CompNet/conivel/tree/gen)  |
|  [LLMaAA: Making Large Language Models as Active Annotators](https://arxiv.org/abs/2310.19596)  |   EMNLP Findings    |  2023-12   |  [Github](https://github.com/ridiculouz/LLMAAA)  |
|  [Prompting ChatGPT in MNER: Enhanced Multimodal Named<br> Entity Recognition with Auxiliary Refined Knowledge](https://arxiv.org/abs/2305.12212)  |   EMNLP Findings    |  2023-12   |  [Github](https://github.com/JinYuanLi0012/PGIM)  |
|  [Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models](https://arxiv.org/abs/2311.08921)  |   Arxiv    |  2023-11   |    |
|  [GPT-NER: Named Entity Recognition via Large Language Models](https://arxiv.org/abs/2304.10428)  |   Arxiv    |  2023-10   |  [Github](https://github.com/ShuheWang1998/GPT-NER)  |
|  [Prompt-NER: Zero-shot Named Entity Recognition in Astronomy Literature via Large Language Models](https://arxiv.org/abs/2310.17892)  |   Arxiv    |  2023-10   |    |
|  [Inspire the Large Language Model by External Knowledge on BioMedical Named Entity Recognition](https://arxiv.org/abs/2309.12278)  |   Arxiv    |  2023-09   |    |
|  [One Model for All Domains: Collaborative Domain-Prefx Tuning for Cross-Domain NER](https://arxiv.org/abs/2301.10410)  |   IJCAI    |  2023-09   |  [Github](https://github.com/zjunlp/DeepKE/tree/main/example/ner/cross)  |
|  [Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and Multimodal Relation Extraction](https://arxiv.org/abs/2306.14122)  |   Arxiv    |  2023-08   |    |
|  [UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition](https://arxiv.org/abs/2308.03279)  |   Arxiv    |  2023-08   |  [Github](https://github.com/universal-ner/universal-ner)  |
|  [Debiasing Generative Named Entity Recognition by Calibrating Sequence Likelihood](https://aclanthology.org/2023.acl-short.98/)  |   ACL Short    |  2023-07   |    |
|  [Entity-to-Text based Data Augmentation for various Named Entity Recognition Tasks](https://aclanthology.org/2023.findings-acl.578/)  |   ACL Findings     |  2023-07   |    |
|  [Large Language Models as Instructors: A Study on Multilingual Clinical Entity Extraction](https://aclanthology.org/2023.bionlp-1.15/)  |   BioNLP    |  2023-07   |  [Github](https://github.com/arkhn/bio-nlp2023)  |
|  [PromptNER : Prompting For Named Entity Recognition](https://arxiv.org/abs/2305.15444)  |   Arxiv    |  2023-06   |  [Github](https://github.com/tricktreat/PromptNER)  |
|  [Does Synthetic Data Generation of LLMs Help Clinical Text Mining?](https://arxiv.org/abs/2303.04360)  |   Arxiv    |  2023-04   |    |
|  [Structured information extraction from complex scientific text with fine-tuned large language models](https://arxiv.org/abs/2212.05238)  |   Arxiv    |  2022-12   |  [Demo](http://www.matscholar.com/info-extraction)  |
|  [LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting](https://aclanthology.org/2022.coling-1.209/) | COLING |  2022-10 |  [Github](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot) |
|  [De-bias for generative extraction in unified NER task](https://aclanthology.org/2022.acl-long.59.pdf)  |   ACL      |  2022-05   |    |
|  [InstructionNER: A Multi-Task Instruction-Based Generative Framework for Few-shot NER](https://arxiv.org/abs/2203.03903) |  Arxiv  |  2022-03 |   |
|  [Document-level Entity-based Extraction as Template Generation](https://aclanthology.org/2021.emnlp-main.426/)  |   EMNLP      |  2021-11   |  [Github](https://github.com/PlusLabNLP/TempGen)  |
|  [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223)  |   ACL      |  2021-08   |  [Github](https://github.com/yhcc/BARTNER)  |
|  [Template-Based Named Entity Recognition Using BART](https://aclanthology.org/2021.findings-acl.161.pdf)  |   ACL Findings    |  2021-08   |  [Github](https://github.com/Nealcly/templateNER)  |


## Relation Extraction 
Models targeting only RE tasks.

### Relation Classification
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [GPT-RE: In-context Learning for Relation Extraction using Large Language Models](https://arxiv.org/abs/2305.02105)  |   EMNLP    |  2023-12   | [Github](https://github.com/YukinoWan/GPT-RE) |
|  [Guideline Learning for In-context Information Extraction](https://arxiv.org/abs/2310.05066)  |   EMNLP    |  2023-12   |  |
|  [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)  |   EMNLP Findings    |  2023-12   | [Github](https://github.com/mayubo2333/LLM-IE) |
|  [LLMaAA: Making Large Language Models as Active Annotators](https://arxiv.org/abs/2310.19596)  |   EMNLP Findings    |  2023-12   | [Github](https://github.com/ridiculouz/LLMAAA) |
|  [Improving Unsupervised Relation Extraction by Augmenting Diverse Sentence Pairs](https://arxiv.org/abs/2312.00552)  |   EMNLP    |  2023-12   | [Github](https://github.com/qingwang-isu/AugURE) |
|  [Revisiting Large Language Models as Zero-shot Relation Extractors](https://arxiv.org/abs/2310.05028)  |   EMNLP Findings    |  2023-12   |  |
|  [Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors](https://aclanthology.org/2023.findings-acl.50.pdf)  |   ACL Findings    |  2023-07   | [Github](https://github.com/OSU-NLP-Group/QA4RE) |
|  [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?](https://arxiv.org/abs/2305.01555)  |   ACL Workshop    |  2023-07   | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/llm/UnleashLLMRE) |
|  [STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models](https://arxiv.org/abs/2305.15090)  |   Arxiv    |  2023-05   |  |
|  [Does Synthetic Data Generation of LLMs Help Clinical Text Mining?](https://arxiv.org/abs/2303.04360)  |   Arxiv    |  2023-04   |  |
|  [REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204/)  |    EMNLP Findings      |   2021-11   | [Github](https://github.com/babelscape/rebel) |


### Relation Triplet
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Document-level Entity-based Extraction as Template Generation](https://aclanthology.org/2021.emnlp-main.426/)  |    EMNLP      | 2021-11    | [Github](https://github.com/PlusLabNLP/TempGen) |

### Relation Strict
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204/)  |    EMNLP Findings      |   2021-11   | [Github](https://github.com/babelscape/rebel) |

## Event Extraction 
Models targeting only EE tasks.

### Event Detection
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Unleash GPT-2 Power for Event Detection](https://aclanthology.org/2021.acl-long.490.pdf)  |    ACL      |  2021-08   | []() |

### Event Argument Extraction
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Context-Aware Prompt for Generation-based Event Argument Extraction with Diffusion Models](https://dl.acm.org/doi/10.1145/3583780.3614820)  |    CIKM      |   2023-10  | []() |
|  [Code4Struct: Code Generation for Few-Shot Event Structure Prediction](https://arxiv.org/abs/2210.12810)  |       ACL   |   2023-07  | [Github](https://github.com/xingyaoww/code4struct) |
|  [Event Extraction as Question Generation and Answering](https://aclanthology.org/2023.acl-short.143.pdf)  |    ACL short      | 2023-07    | [Github](https://github.com/dataminr-ai/Event-Extraction-as-Question-Generation-and-Answering) |
|  [Prompt for extraction? PAIE: prompting argument interaction for event argument extraction](https://aclanthology.org/2022.acl-long.466/)  |        ACL       |  2022-05   | [Github](https://github.com/mayubo2333/PAIE) |

### Event Detection & Argument Extraction
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Guideline Learning for In-context Information Extraction](https://arxiv.org/abs/2310.05066)  |   EMNLP    |  2023-12   |  |
|  [DemoSG: Demonstration-enhanced Schema-guided Generation for Low-resource Event Extraction](https://aclanthology.org/2023.findings-emnlp.121) |     EMNLP Findings     | 2023-12    | [Github](https://github.com/GangZhao98/DemoSG) |
|  [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)  |     EMNLP Findings     | 2023-12    | [Github](https://github.com/mayubo2333/LLM-IE) |
|  [DICE: Data-Efficient Clinical Event Extraction with Generative Models](https://aclanthology.org/2023.acl-long.886.pdf)  |      ACL    |  2023-07   | [Github](https://github.com/derekmma/DICE) |
|  [A Monte Carlo Language Model Pipeline for Zero-Shot Sociopolitical Event Extraction](https://arxiv.org/abs/2305.15051)   |    Arxiv      |  2023-05    | []() |
|  [STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models](https://arxiv.org/abs/2305.15090)  |     Arxiv     |  2023-05   | []() |
|  [DEGREE: A Data-Efficient Generative Event Extraction Model](https://aclanthology.org/2022.naacl-main.138/)  |      NAACL    |   2022-07  | [Github](https://github.com/PlusLabNLP/DEGREE) |
|  [ClarET: Pre-training a correlation-aware context-to-event transformer for event-centric generation and classification](https://aclanthology.org/2022.acl-long.183/)  |   ACL       |    2022-05 | [Github](https://github.com/yczhou001/ClarET) |
|  [Dynamic prefix-tuning for generative template-based event extraction](https://aclanthology.org/2022.acl-long.358.pdf)  |   ACL       |  2022-05   | []() |
|  [Text2event: Controllable sequence-to- structure generation for end-to-end event extraction](https://arxiv.org/abs/2106.09232)  |        ACL     |  2021-08     | [Github](https://github.com/luyaojie/text2event) |
|  [Document-level event argument extraction by conditional generation](https://aclanthology.org/2021.naacl-main.69.pdf)  |    NAACL      | 2021-06    | [Github](https://github.com/raspberryice/gen-arg) |


## Universal Information Extraction
Unified models targeting multiple IE tasks.
### NL-LLMs based
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Set Learning for Generative Information Extraction](https://aclanthology.org/2023.emnlp-main.806.pdf)  |    EMNLP      |   2023-12  | []() |
|  [GIELLM: Japanese General Information Extraction Large Language Model Utilizing Mutual Reinforcement Effect](https://arxiv.org/abs/2311.06838)  |   Arxiv       |  2023-11   | []() |
|  [InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction](https://arxiv.org/abs/2304.08085)  |    Arxiv      |  2023-04   | [Github](https://github.com/BeyonderXX/InstructUIE) |
| [Zero-Shot Information Extraction via Chatting with ChatGPT](https://arxiv.org/abs/2302.10205)   |      Arxiv      |  2023-02    | [Github](https://github.com/cocacola-lab/ChatIE) |
|  [GenIE: Generative Information Extraction](https://aclanthology.org/2022.naacl-main.342.pdf)  |     NAACL     |   2022-07  | [Github](https://github.com/epfl-dlab/GenIE) |
|  [DEEPSTRUCT: Pretraining of Language Models for Structure Prediction](https://aclanthology.org/2022.findings-acl.67/)  |      ACL Findings    |   2022-05  | [Github](https://github.com/wang-research-lab/deepstruct) |
|  [Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395/)  |      ACL       |  2022-05     | [Github](https://github.com/yhcc/BARTABSA) |
|  [Structured prediction as translation between augmented natural languages](https://arxiv.org/abs/2101.05779)  |    ICLR      | 2021-01    | [Github](https://github.com/amazon-science/tanl) |

### Code-LLMs based
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction](https://arxiv.org/abs/2310.03668)  |     Arxiv     |  2023-12   | [Github](https://github.com/hitz-zentroa/GoLLIE) |
|  [Retrieval-Augmented Code Generation for Universal Information Extraction](https://arxiv.org/abs/2311.02962)  |      Arxiv    |   2023-11  | []() |
|  [CODEIE: Large Code Generation Models are Better Few-Shot Information Extractors](https://arxiv.org/abs/2305.05711)  |       ACL   |   2023-07 | [Github](https://github.com/artpli/CodeIE) |
|  [CodeKGC: Code Language Model for Generative Knowledge Graph Construction](https://arxiv.org/abs/2304.09048)  |    Arxiv      |   2023-04  | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/llm/CodeKGC) |


# Learning Paradigms
A taxonomy by Learning Paradigms.

## Supervised Fine-tuning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction](https://arxiv.org/abs/2310.03668)  |     Arxiv     |  2023-12   | [Github](https://github.com/hitz-zentroa/GoLLIE) |
|  [Set Learning for Generative Information Extraction](https://aclanthology.org/2023.emnlp-main.806.pdf)  |    EMNLP      |   2023-12  | []() |
|  [DemoSG: Demonstration-enhanced Schema-guided Generation for Low-resource Event Extraction](https://aclanthology.org/2023.findings-emnlp.121) |     EMNLP Findings     | 2023-12    | [Github](https://github.com/GangZhao98/DemoSG) |
|  [Calibrated Seq2seq Models for Efficient and Generalizable Ultra-fine Entity Typing](https://aclanthology.org/2023.findings-emnlp.1040/)  |     EMNLP Findings    |   2023-12  | []() |
|  [GIELLM: Japanese General Information Extraction Large Language Model Utilizing Mutual Reinforcement Effect](https://arxiv.org/abs/2311.06838)  |   Arxiv       |  2023-11   | []() |
|  [Context-Aware Prompt for Generation-based Event Argument Extraction with Diffusion Models](https://dl.acm.org/doi/10.1145/3583780.3614820)  |    CIKM      |   2023-10  | []() |
|  [UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition](https://arxiv.org/abs/2308.03279)  |   Arxiv       |   2023-08  | [Github](https://github.com/universal-ner/universal-ner) |
|  [Debiasing Generative Named Entity Recognition by Calibrating Sequence Likelihood](https://aclanthology.org/2023.acl-short.98/)  |     ACL short    |  2023-07   | []() |
|  [DICE: Data-Efficient Clinical Event Extraction with Generative Models](https://aclanthology.org/2023.acl-long.886.pdf)  |      ACL    |  2023-07   | [Github](https://github.com/derekmma/DICE) |
|  [Event Extraction as Question Generation and Answering](https://aclanthology.org/2023.acl-short.143.pdf)  |    ACL short      | 2023-07    | [Github](https://github.com/dataminr-ai/Event-Extraction-as-Question-Generation-and-Answering) |
|  [InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction](https://arxiv.org/abs/2304.08085)  |    Arxiv      |  2023-04   | [Github](https://github.com/BeyonderXX/InstructUIE) |
|  [Structured information extraction from complex scientific text with fine-tuned large language models](https://arxiv.org/abs/2212.05238)  |    Arxiv      |  2022-12   | [Demo](http://www.matscholar.com/info-extraction) |
|  [Generative Entity Typing with Curriculum Learning](https://arxiv.org/abs/2210.02914)  |     EMNLP     |  2022-12   | [Github](https://github.com/siyuyuan/GET) |
|  [LasUIE: Unifying Information Extraction with Latent Adaptive Structure-aware Generative Language Model](https://openreview.net/pdf?id=a8qX5RG36jd)  |     NeurIPS     |   2022-10  | [Github](https://github.com/ChocoWu/LasUIE) |
|  [LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting](https://aclanthology.org/2022.coling-1.209/) | COLING |  2022-10 |  [Github](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot) |
|  [GenIE: Generative Information Extraction](https://aclanthology.org/2022.naacl-main.342.pdf)  |     NAACL     |   2022-07  | [Github](https://github.com/epfl-dlab/GenIE) |
|  [DEGREE: A Data-Efficient Generative Event Extraction Model](https://aclanthology.org/2022.naacl-main.138/)  |      NAACL    |   2022-07  | [Github](https://github.com/PlusLabNLP/DEGREE) |
|  [ClarET: Pre-training a correlation-aware context-to-event transformer for event-centric generation and classification](https://aclanthology.org/2022.acl-long.183/)  |   ACL       |    2022-05 | [Github](https://github.com/yczhou001/ClarET) |
|  [DEEPSTRUCT: Pretraining of Language Models for Structure Prediction](https://aclanthology.org/2022.findings-acl.67/)  |      ACL Findings    |   2022-05  | [Github](https://github.com/wang-research-lab/deepstruct) |
|  [Dynamic prefix-tuning for generative template-based event extraction](https://aclanthology.org/2022.acl-long.358.pdf)  |   ACL       |  2022-05   | []() |
|  [Prompt for extraction? PAIE: prompting argument interaction for event argument extraction](https://aclanthology.org/2022.acl-long.466/)  |        ACL       |  2022-05   | [Github](https://github.com/mayubo2333/PAIE) |
|  [Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395/)  |      ACL       |  2022-05     | [Github](https://github.com/yhcc/BARTABSA) |
|  [De-bias for generative extraction in unified NER task](https://aclanthology.org/2022.acl-long.59.pdf)  |       ACL       |  2022-05    | []() |
|  [Document-level Entity-based Extraction as Template Generation](https://aclanthology.org/2021.emnlp-main.426/)  |    EMNLP      | 2021-11    | [Github](https://github.com/PlusLabNLP/TempGen) |
|  [REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204/)  |    EMNLP Findings      |   2021-11   | [Github](https://github.com/babelscape/rebel) |
|  [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223)  |   ACL       |   2021-08  | [Github](https://github.com/yhcc/BARTNER) |
|  [Template-Based Named Entity Recognition Using BART](https://aclanthology.org/2021.findings-acl.161.pdf)  |    ACL Findings     |  2021-08   | [Github](https://github.com/Nealcly/templateNER) |
|  [Text2event: Controllable sequence-to- structure generation for end-to-end event extraction](https://arxiv.org/abs/2106.09232)  |        ACL     |  2021-08     | [Github](https://github.com/luyaojie/text2event) |
|  [Document-level event argument extraction by conditional generation](https://aclanthology.org/2021.naacl-main.69.pdf)  |    NAACL      | 2021-06    | [Github](https://github.com/raspberryice/gen-arg) |
|  [Structured prediction as translation between augmented natural languages](https://arxiv.org/abs/2101.05779)  |    ICLR      | 2021-01    | [Github](https://github.com/amazon-science/tanl) |

## Few-shot
### Few-shot Fine-tuning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [DemoSG: Demonstration-enhanced Schema-guided Generation for Low-resource Event Extraction](https://aclanthology.org/2023.findings-emnlp.121) |     EMNLP Findings     | 2023-12    | [Github](https://github.com/GangZhao98/DemoSG) |
|  [One Model for All Domains: Collaborative Domain-Prefx Tuning for Cross-Domain NER](https://arxiv.org/abs/2301.10410)  |    IJCAI      | 2023-09    | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/ner/cross)  |
|  [LightNER: A Lightweight Tuning Paradigm for Low-resource NER via Pluggable Prompting](https://aclanthology.org/2022.coling-1.209/) | COLING |  2022-10 |  [Github](https://github.com/zjunlp/DeepKE/tree/main/example/ner/few-shot) |
|  [Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395/)  |      ACL       |  2022-05     | [Github](https://github.com/yhcc/BARTABSA) |
|  [InstructionNER: A Multi-Task Instruction-Based Generative Framework for Few-shot NER](https://arxiv.org/abs/2203.03903) |  Arxiv  |  2022-03 |   |
|  [Template-Based Named Entity Recognition Using BART](https://aclanthology.org/2021.findings-acl.161.pdf)  |    ACL Findings     |  2021-08   | [Github](https://github.com/Nealcly/templateNER) |
|  [Structured prediction as translation between augmented natural languages](https://arxiv.org/abs/2101.05779)  |    ICLR      | 2021-01    | [Github](https://github.com/amazon-science/tanl) |

### In-Context Learning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [GPT-RE: In-context Learning for Relation Extraction using Large Language Models](https://arxiv.org/abs/2305.02105)  |     EMNLP     |   2023-12  | [Github](https://github.com/YukinoWan/GPT-RE) |
|  [Guideline Learning for In-context Information Extraction](https://arxiv.org/abs/2310.05066)  |  EMNLP     |   2023-12  | []() |
|  [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)  |     EMNLP Findings     | 2023-12    | [Github](https://github.com/mayubo2333/LLM-IE) |
|  [Retrieval-Augmented Code Generation for Universal Information Extraction](https://arxiv.org/abs/2311.02962)  |      Arxiv    |   2023-11  | []() |
|  [Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models](https://arxiv.org/abs/2311.08921)  |Arxiv    |   2023-11   | []() |
|  [GPT-NER: Named Entity Recognition via Large Language Models](https://arxiv.org/abs/2304.10428)  |    Arxiv    |   2023-10   | [Github](https://github.com/ShuheWang1998/GPT-NER) |
|  [Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors](https://aclanthology.org/2023.findings-acl.50.pdf)  |     ACL Findings     |   2023-07  | [Github](https://github.com/OSU-NLP-Group/QA4RE) |
|  [Code4Struct: Code Generation for Few-Shot Event Structure Prediction](https://arxiv.org/abs/2210.12810)  |       ACL   |   2023-07  | [Github](https://github.com/xingyaoww/code4struct) |
|  [CODEIE: Large Code Generation Models are Better Few-Shot Information Extractors](https://arxiv.org/abs/2305.05711)  |       ACL   |   2023-07 | [Github](https://github.com/artpli/CodeIE) |
|  [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?](https://arxiv.org/abs/2305.01555)  |    ACL Workshop      |  2023-07    | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/llm/UnleashLLMRE) |
|  [PromptNER : Prompting For Named Entity Recognition](https://arxiv.org/abs/2305.15444)  |      Arxiv    |  2023-06   | [Github](https://github.com/tricktreat/PromptNER) |
|  [CodeKGC: Code Language Model for Generative Knowledge Graph Construction](https://arxiv.org/abs/2304.09048)  |    Arxiv      |   2023-04  | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/llm/CodeKGC) |

## Zero-shot
### Zero-shot Prompting
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
| [Improving Unsupervised Relation Extraction by Augmenting Diverse Sentence Pairs](https://arxiv.org/abs/2312.00552)   |   EMNLP       |    2023-12  | [Github](https://github.com/qingwang-isu/AugURE) |
| [Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models](https://arxiv.org/abs/2311.08921)   |   Arxiv       |  2023-11    | []() |
|  [Prompt-NER: Zero-shot Named Entity Recognition in Astronomy Literature via Large Language Models](https://arxiv.org/abs/2310.17892)    | Arxiv  |   2023-10  |  |
| [Revisiting Large Language Models as Zero-shot Relation Extractors](https://arxiv.org/abs/2310.05028)   |     EMNLP Findings     |  2023-10    | []() |
| [Aligning Instruction Tasks Unlocks Large Language Models as Zero-Shot Relation Extractors](https://aclanthology.org/2023.findings-acl.50.pdf)   |     ACL Findings     |  2023-07    | [Github](https://github.com/OSU-NLP-Group/QA4RE) |
| [Code4Struct: Code Generation for Few-Shot Event Structure Prediction](https://arxiv.org/abs/2210.12810)   |    ACL      |   2023-07   | [Github](https://github.com/xingyaoww/code4struct) |
| [A Monte Carlo Language Model Pipeline for Zero-Shot Sociopolitical Event Extraction](https://arxiv.org/abs/2305.15051)   |    Arxiv      |  2023-05    | []() |
| [CodeKGC: Code Language Model for Generative Knowledge Graph Construction](https://arxiv.org/abs/2304.09048)   |        Arxiv      |  2023-04    | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/llm/CodeKGC) |
| [Zero-Shot Information Extraction via Chatting with ChatGPT](https://arxiv.org/abs/2302.10205)   |      Arxiv      |  2023-02    | [Github](https://github.com/cocacola-lab/ChatIE) |

### Cross-Domain Learning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction](https://arxiv.org/abs/2310.03668)  |     Arxiv     |  2023-12   | [Github](https://github.com/hitz-zentroa/GoLLIE) |
|  [UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition](https://arxiv.org/abs/2308.03279)  |  Arxiv        |   2023-08  | [Github](https://github.com/universal-ner/universal-ner) |
|  [InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction](https://arxiv.org/abs/2304.08085)  |     Arxiv     |  2023-04   | [Github](https://github.com/BeyonderXX/InstructUIE) |
|  [DEEPSTRUCT: Pretraining of Language Models for Structure Prediction](https://aclanthology.org/2022.findings-acl.67/)  |   ACL Findings       |  2022-05   | [Github](https://github.com/wang-research-lab/deepstruct) |
|  [Multilingual generative language models for zero-shot cross-lingual event argument extraction](https://aclanthology.org/2022.acl-long.317.pdf)  |      ACL   |  2022-05    | [Github](https://github.com/PlusLabNLP/X-Gear) |

### Cross-Type Learning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Document-level event argument extraction by conditional generation](https://aclanthology.org/2021.naacl-main.69.pdf)  |     NAACL     |   2021-06  | [Github](https://github.com/raspberryice/gen-arg) |


## Data Augmentation
### Data Annotation
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [LLMaAA: Making Large Language Models as Active Annotators](https://arxiv.org/abs/2310.19596)  |   EMNLP Findings       |  2023-12   | [Github](https://github.com/ridiculouz/LLMAAA) |
|  [Improving Unsupervised Relation Extraction by Augmenting Diverse Sentence Pairs](https://arxiv.org/abs/2312.00552)  |      EMNLP    |  2023-12   | [Github](https://github.com/qingwang-isu/AugURE) |
|  [Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models](https://arxiv.org/pdf/2311.07314v1.pdf)  |    Arxiv      |  2023-11   | [Github](https://github.com/bigai-nlco/DocGNRE) |
|  [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?](https://arxiv.org/abs/2305.01555)  |     ACL Workshop     |   2023-07  | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/llm/UnleashLLMRE) |
|  [Large Language Models as Instructors: A Study on Multilingual Clinical Entity Extraction](https://aclanthology.org/2023.bionlp-1.15/)  |   bioNLP Workshop       |  2023-07   | [Github](https://github.com/arkhn/bio-nlp2023) |
|  [Does Synthetic Data Generation of LLMs Help Clinical Text Mining?](https://arxiv.org/abs/2303.04360)  |     Arxiv     |   2023-04  | []() |
|  [Unleash GPT-2 Power for Event Detection](https://aclanthology.org/2021.acl-long.490.pdf)  |    ACL      |  2021-08   | []() |

### Knowledge Retrieval 
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Learning to Rank Context for Named Entity Recognition Using a Synthetic Dataset](https://arxiv.org/abs/2310.10118)  |    EMNLP      |  2023-12   | [Github](https://github.com/CompNet/conivel/tree/gen) |
|  [Prompting ChatGPT in MNER: Enhanced Multimodal Named Entity Recognition with Auxiliary <br>Refined Knowledge](https://arxiv.org/abs/2305.12212)  |  EMNLP Findings        |  2023-12   | [Github](https://github.com/JinYuanLi0012/PGIM) |
|  [Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and <br>Multimodal Relation Extraction](https://arxiv.org/abs/2306.14122)  |     Arxiv     |  2023-08   | []() |

### Inverse Generation
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Exploiting Asymmetry for Synthetic Training Data Generation: SynthIE and the Case of Information Extraction](https://arxiv.org/abs/2303.04132)  |   EMNLP       |  2023-12   | [Github](https://github.com/epfl-dlab/SynthIE) |
|  [Entity-to-Text based Data Augmentation for various Named Entity Recognition Tasks](https://aclanthology.org/2023.findings-acl.578/)  |     ACL Findings     |   2023-07  | []() |
|  [Event Extraction as Question Generation and Answering](https://aclanthology.org/2023.acl-short.143.pdf)  |   ACL Short       |  2023-07   | [Github](https://github.com/dataminr-ai/Event-Extraction-as-Question-Generation-and-Answering) |
|  [STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models](https://arxiv.org/abs/2305.15090)  |     Arxiv     |  2023-05   | []() |


# Specific Domain

|  Paper  |  Domain |   Venue    |   Date  | Code |
| :----- | :--------------: | :-------: | :---------: |:---------: |
|  [Prompting ChatGPT in MNER: Enhanced Multimodal Named Entity Recognition with<br> Auxiliary Refined Knowledge](https://arxiv.org/abs/2305.12212)  |   Multimodal      | ENMLP Findings  |  2023-12  | [Github](https://github.com/JinYuanLi0012/PGIM) |
|  [In-context Learning for Few-shot Multimodal Named Entity Recognition](https://aclanthology.org/2023.findings-emnlp.196/)  |     Multimodal    | ENMLP Findings  | 2023-12   |  |
|  [PolyIE: A Dataset of Information Extraction from Polymer Material Scientific Literature](https://arxiv.org/abs/2311.07715)  |     Polymer Material    | Arxiv  |   2023-11 | [Github](https://github.com/jerry3027/PolyIE) |
|  [Prompt-NER: Zero-shot Named Entity Recognition in Astronomy Literature via Large Language Models](https://arxiv.org/abs/2310.17892)  |     Astronomical    | Arxiv  |   2023-10  |  |
|  [Inspire the Large Language Model by External Knowledge on BioMedical Named Entity Recognition](https://arxiv.org/abs/2309.12278)  |    Biomedical     |  Arxiv  |   2023-09  | |
|  [Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and Multimodal Relation Extraction](https://arxiv.org/abs/2306.14122)  |    Multimodal     |  Arxiv  |   2023-08  | |
|  [DICE: Data-Efficient Clinical Event Extraction with Generative Models](https://aclanthology.org/2023.acl-long.886.pdf)  |   Clinical      |  ACL | 2023-07   | [Github](https://github.com/derekmma/DICE) |
|  [How far is Language Model from 100% Few-shot Named Entity Recognition in Medical Domain](https://arxiv.org/abs/2307.00186)  |     Medical    | Arxiv  |  2023-07  | [Github](https://github.com/ToneLi/RT-Retrieving-and-Thinking) |
|  [Large Language Models as Instructors: A Study on Multilingual Clinical Entity Extraction](https://aclanthology.org/2023.bionlp-1.15/)  |     Multilingual / Clinical      | BioNLP  |  2023-07  | [Github](https://github.com/arkhn/bio-nlp2023) |
|  [Does Synthetic Data Generation of LLMs Help Clinical Text Mining?](https://arxiv.org/abs/2303.04360)  |   Clinical      | Arxiv  |  2023-04  |  |
|  [Yes but.. Can ChatGPT Identify Entities in Historical Documents](https://arxiv.org/abs/2303.17322)  |    Historical    |  JCDL   |  2023-03  |  |
|  [Zero-shot Clinical Entity Recognition using ChatGPT](https://arxiv.org/abs/2303.16416)  |   Clinical |      Arxiv   |  2023-03    |  |
|  [Structured information extraction from complex scientific text with fine-tuned large language models](https://arxiv.org/abs/2212.05238)  |   Scientific  |      Arxiv   |  2022-12   | [Demo](http://www.matscholar.com/info-extraction) |
|  [Multilingual generative language models for zero-shot cross-lingual event argument extraction](https://aclanthology.org/2022.acl-long.317.pdf)  |   Multilingual |      ACL   |  2022-05    | [Github](https://github.com/PlusLabNLP/X-Gear) |

# Evaluation and Analysis
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|  [Empirical Study of Zero-Shot NER with ChatGPT](https://arxiv.org/abs/2310.10035)  |   EMNLP       |  2023-12   | [Github](https://github.com/Emma1066/Zero-Shot-NER-with-ChatGPT) |
|  [NERetrieve: Dataset for Next Generation Named Entity Recognition and Retrieval](https://arxiv.org/abs/2310.14282)  |   EMNLP Findings       |  2023-12   | [Github](https://github.com/katzurik/NERetrieve) |
|  [Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction](https://aclanthology.org/2023.emnlp-main.360/)  |  EMNLP    |   2023-12  | [Github](https://github.com/qijimrc/ROBUST) |
|  [PolyIE: A Dataset of Information Extraction from Polymer Material Scientific Literature](https://arxiv.org/abs/2311.07715)  |  Arxiv    |   2023-11  | [Github](https://github.com/jerry3027/PolyIE) |
|  [XNLP: An Interactive Demonstration System for Universal Structured NLP](https://arxiv.org/abs/2308.01846)  |  Arxiv    |  2023-08   | [Demo](http://xnlp.haofei.vip/) |
|  [A Zero-shot and Few-shot Study of Instruction-Finetuned Large Language Models Applied to Clinical and Biomedical Tasks](https://arxiv.org/abs/2307.12114)  |   Arxiv    |  2023-07  |  |
|  [How far is Language Model from 100% Few-shot Named Entity Recognition in Medical Domain](https://arxiv.org/abs/2307.00186)  |     Arxiv    |  2023-07   | [Github](https://github.com/ToneLi/RT-Retrieving-and-Thinking) |
|  [Revisiting Relation Extraction in the era of Large Language Models](https://aclanthology.org/2023.acl-long.868.pdf)  |   ACL   |  2023-07  | [Github](https://sominw.com/ACL23LLMs) |
|  [Zero-shot Temporal Relation Extraction with ChatGPT](https://aclanthology.org/2023.bionlp-1.7/)  |  BioNLP    |  2023-07   |  |
|  [InstructIE: A Chinese Instruction-based Information Extraction Dataset](https://arxiv.org/abs/2305.11527)  |   Arxiv   |  2023-05  | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/llm) |
|  [Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors](https://arxiv.org/abs/2305.14450)  |   Arxiv   |  2023-05    | [Github](https://github.com/FreedomIntelligence/Evaluation-of-ChatGPT-on-Information-Extraction) |
|  [Evaluating ChatGPT's Information Extraction Capabilities: An Assessment of Performance, Explainability, Calibration, and Faithfulness](https://arxiv.org/abs/2304.11633)  |   Arxiv   |  2023-04   | [Github](https://github.com/pkuserc/ChatGPT_for_IE) |
|  [Exploring the Feasibility of ChatGPT for Event Extraction](https://arxiv.org/abs/2303.03836)  |     Arxiv   |  2023-03   |  |
|  [Yes but.. Can ChatGPT Identify Entities in Historical Documents](https://arxiv.org/abs/2303.17322)  |   JCDL   |  2023-03   |  |
|  [Zero-shot Clinical Entity Recognition using ChatGPT](https://arxiv.org/abs/2303.16416)  |      Arxiv   |  2023-03    |  |
|  [Thinking about GPT-3 In-Context Learning for Biomedical IE? Think Again](https://aclanthology.org/2022.findings-emnlp.329/)  |   EMNLP Findings   |   2022-12  | [Github](https://github.com/dki-lab/few-shot-bioIE) |
|  [Large Language Models are Few-Shot Clinical Information Extractors](https://arxiv.org/abs/2205.12689)  |   EMNLP   |  2022-12   | [Huggingface](https://huggingface.co/datasets/mitclinicalml/clinical-ie) |

# Datasets
\* denotes the dataset is multimodal. # refers to the number of categories or sentences.
<br>
The link will be updated soon~

<table>
    <thead>
        <tr>
            <th align="center">Task</th>
            <th align="center">Dataset</th>
            <th align="center">Domain</th>
            <th align="center">#Class</th>
            <th align="center">#Train</th>
            <th align="center">#Val</th>
            <th align="center">#Test</th>
            <th align="center">Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center" rowspan="32" ><strong>NER</strong></td>
            <td align="center">ACE04</td>
            <td align="center">News</td>
            <td align="center">7</td>
            <td align="center">6202</td>
            <td align="center">745</td>
            <td align="center">812</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2005T09">Link</a></td>
        </tr>
        <tr>
            <td align="center">ACE05</td>
            <td align="center">News</td>
            <td align="center">7</td>
            <td align="center">7299</td>
            <td align="center">971</td>
            <td align="center">1060</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">BC5CDR</td>
            <td align="center">Biomedical</td>
            <td align="center">2</td>
            <td align="center">4560</td>
            <td align="center">4581</td>
            <td align="center">4797</td>
            <td align="center"><a href="https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/">Link</a></td>
        </tr>
        <tr>
            <td align="center">Broad Twitter Corpus</td>
            <td align="center">Social Media</td>
            <td align="center">3</td>
            <td align="center">6338</td>
            <td align="center">1001</td>
            <td align="center">2000</td>
            <td align="center"><a href="https://github.com/GateNLP/broad_twitter_corpus?">Link</a></td>
        </tr>
        <tr>
            <td align="center">CADEC</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">5340</td>
            <td align="center">1097</td>
            <td align="center">1160</td>
            <td align="center"><a href="https://data.csiro.au/collection/csiro:10948?v=3&d=true">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLL03</td>
            <td align="center">News</td>
            <td align="center">4</td>
            <td align="center">14041</td>
            <td align="center">3250</td>
            <td align="center">3453</td>
            <td align="center"><a href="https://www.clips.uantwerpen.be/conll2003/ner/">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLLpp</td>
            <td align="center">News</td>
            <td align="center">4</td>
            <td align="center">14041</td>
            <td align="center">3250</td>
            <td align="center">3453</td>
            <td align="center"><a href="https://github.com/ZihanWangKi/CrossWeigh">Link</a></td>
        </tr>
        <tr>
            <td align="center">CrossNER-AI</td>
            <td align="center">Artificial Intelligence</td>
            <td align="center">14</td>
            <td align="center">100</td>
            <td align="center">350</td>
            <td align="center">431</td>
            <td align="center" rowspan="5" ><a href="https://github.com/zliucr/CrossNER">Link</a></td>
        </tr>
        <tr>
            <td align="center">CrossNER-Literature</td>
            <td align="center">Literary</td>
            <td align="center">12</td>
            <td align="center">100</td>
            <td align="center">400</td>
            <td align="center">416</td>
        </tr>
        <tr>
            <td align="center">CrossNER-Music</td>
            <td align="center">Musical</td>
            <td align="center">13</td>
            <td align="center">100</td>
            <td align="center">380</td>
            <td align="center">465</td>
        </tr>
        <tr>
            <td align="center">CrossNER-Politics</td>
            <td align="center">Political</td>
            <td align="center">9</td>
            <td align="center">199</td>
            <td align="center">540</td>
            <td align="center">650</td>
        </tr>
        <tr>
            <td align="center">CrossNER-Science</td>
            <td align="center">Scientific</td>
            <td align="center">17</td>
            <td align="center">200</td>
            <td align="center">450</td>
            <td align="center">543</td>
        </tr>
        <tr>
            <td align="center">FabNER</td>
            <td align="center">Scientific</td>
            <td align="center">12</td>
            <td align="center">9435</td>
            <td align="center">2182</td>
            <td align="center">2064</td>
            <td align="center"><a href="https://huggingface.co/datasets/DFKI-SLT/fabner">Link</a></td>
        </tr>
        <tr>
            <td align="center">Few-NERD</td>
            <td align="center">General</td>
            <td align="center">66</td>
            <td align="center">131767</td>
            <td align="center">18824</td>
            <td align="center">37468</td>
            <td align="center"><a href="https://github.com/thunlp/Few-NERD">Link</a></td>
        </tr>
        <tr>
            <td align="center">FindVehicle</td>
            <td align="center">Traffic</td>
            <td align="center">21</td>
            <td align="center">21565</td>
            <td align="center">20777</td>
            <td align="center">20777</td>
            <td align="center"><a href="https://github.com/GuanRunwei/VehicleFinder-CTIM">Link</a></td>
        </tr>
        <tr>
            <td align="center">GENIA</td>
            <td align="center">Biomedical</td>
            <td align="center">5</td>
            <td align="center">15023</td>
            <td align="center">1669</td>
            <td align="center">1854</td>
            <td align="center"><a href="https://github.com/ljynlp/W2NER?tab=readme-ov-file#3-dataset">Link</a></td>
        </tr>
        <tr>
            <td align="center">HarveyNER</td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">3967</td>
            <td align="center">1301</td>
            <td align="center">1303</td>
            <td align="center"><a href="https://github.com/brickee/HarveyNER">Link</a></td>
        </tr>
        <tr>
            <td align="center">MIT-Movie</td>
            <td align="center">Social Media</td>
            <td align="center">12</td>
            <td align="center">9774</td>
            <td align="center">2442</td>
            <td align="center">2442</td>
            <td align="center"><a href="https://tianchi.aliyun.com/dataset/145106">Link</a></td>
        </tr>
        <tr>
            <td align="center">MIT-Restaurant</td>
            <td align="center">Social Media</td>
            <td align="center">8</td>
            <td align="center">7659</td>
            <td align="center">1520</td>
            <td align="center">1520</td>
            <td align="center"><a href="https://tianchi.aliyun.com/dataset/145105">Link</a></td>
        </tr>
        <tr>
            <td align="center">MultiNERD</td>
            <td align="center">Wikipedia</td>
            <td align="center">16</td>
            <td align="center">134144</td>
            <td align="center">10000</td>
            <td align="center">10000</td>
            <td align="center"><a href="https://github.com/Babelscape/multinerd">Link</a></td>
        </tr>
        <tr>
            <td align="center">NCBI</td>
            <td align="center">Biomedical</td>
            <td align="center">4</td>
            <td align="center">5432</td>
            <td align="center">923</td>
            <td align="center">940</td>
            <td align="center"><a href="https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/">Link</a></td>
        </tr>
        <tr>
            <td align="center">OntoNotes 5.0</td>
            <td align="center">General</td>
            <td align="center">18</td>
            <td align="center">59924</td>
            <td align="center">8528</td>
            <td align="center">8262</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2013T19">Link</a></td>
        </tr>
        <tr>
            <td align="center">ShARe13</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">8508</td>
            <td align="center">12050</td>
            <td align="center">9009</td>
            <td align="center"><a href="https://physionet.org/content/shareclefehealth2013/1.0/">Link</a></td>
        </tr>
        <tr>
            <td align="center">ShARe14</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">17404</td>
            <td align="center">1360</td>
            <td align="center">15850</td>
            <td align="center"><a href="https://physionet.org/content/shareclefehealth2014task2/1.0/">Link</a></td>
        </tr>
        <tr>
            <td align="center">SNAP<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">4290</td>
            <td align="center">1432</td>
            <td align="center">1459</td>
            <td align="center"><a href="https://www.modelscope.cn/datasets/caijiong_sijun/MoRE-processed-data/files">Link</a></td>
        </tr>
        <tr>
            <td align="center">Temporal Twitter Corpus (TTC)</td>
            <td align="center">Social Meida</td>
            <td align="center">3</td>
            <td align="center">10000</td>
            <td align="center">500</td>
            <td align="center">1500</td>
            <td aligh="center"><a href="https://github.com/shrutirij/temporal-twitter-corpus">Link</a></td>
        </tr>
        <tr>
            <td align="center">Tweebank-NER</td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">1639</td>
            <td align="center">710</td>
            <td align="center">1201</td>
            <td aligh="center"><a href="https://github.com/mit-ccc/TweebankNLP">Link</a></td>
        </tr>
        <tr>
            <td align="center">Twitter2015<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">4000</td>
            <td align="center">1000</td>
            <td align="center">3357</td>
            <td align="center"><a href="https://www.modelscope.cn/datasets/caijiong_sijun/MoRE-processed-data/files">Link</a></td>
        </tr>
        <tr>
            <td align="center">Twitter2017<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">4</td>
            <td align="center">3373</td>
            <td align="center">723</td>
            <td align="center">723</td>
            <td align="center"><a href="https://www.modelscope.cn/datasets/caijiong_sijun/MoRE-processed-data/files">Link</a></td>
        </tr>
        <tr>
            <td align="center">TwitterNER7</td>
            <td align="center">Social Media</td>
            <td align="center">7</td>
            <td align="center">7111</td>
            <td align="center">886</td>
            <td align="center">576</td>
            <td aligh="center"><a href="https://huggingface.co/datasets/tner/tweetner7">Link</a></td>
        </tr>
        <tr>
            <td align="center">WikiDiverse<sup>*<sup></td>
            <td align="center">News</td>
            <td align="center">13</td>
            <td align="center">6312</td>
            <td align="center">755</td>
            <td align="center">757</td>
            <td aligh="center"><a href="https://github.com/wangxw5/wikidiverse?tab=readme-ov-file#get-the-data">Link</a></td>
        </tr>
        <tr>
            <td align="center">WNUT2017</td>
            <td align="center">Social Media</td>
            <td align="center">6</td>
            <td align="center">3394</td>
            <td align="center">1009</td>
            <td align="center">1287</td>
            <td aligh="center"><a href="https://tianchi.aliyun.com/dataset/144349">Link</a></td>
        </tr>
        <tr>
            <td align="center" rowspan="11"><strong>RE</strong></td>
            <td align="center">ACE05</td>
            <td align="center">News</td>
            <td align="center">7</td>
            <td align="center">10051</td>
            <td align="center">2420</td>
            <td align="center">2050</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">ADE</td>
            <td align="center">Biomedical</td>
            <td align="center">1</td>
            <td align="center">3417</td>
            <td align="center">427</td>
            <td align="center">428</td>
            <td align="center"><a href="http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/">Link</a></td>
        </tr>
        <tr>
            <td align="center">CoNLL04</td>
            <td align="center">News</td>
            <td align="center">5</td>
            <td align="center">922</td>
            <td align="center">231</td>
            <td align="center">288</td>
            <td align="center"><a href="http://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/">Link</a></td>
        </tr>
        <tr>
            <td align="center">DocRED</td>
            <td align="center">Wikipedia</td>
            <td align="center">96</td>
            <td align="center">3008</td>
            <td align="center">300</td>
            <td align="center">700</td>
            <td align="center"><a href="https://github.com/thunlp/DocRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">MNRE<sup>*<sup></td>
            <td align="center">Social Media</td>
            <td align="center">23</td>
            <td align="center">12247</td>
            <td align="center">1624</td>
            <td align="center">1614</td>
            <td align="center"><a href="https://github.com/thecharm/MNRE">Link</a></td>
        </tr>
        <tr>
            <td align="center">NYT</td>
            <td align="center">News</td>
            <td align="center">24</td>
            <td align="center">56196</td>
            <td align="center">5000</td>
            <td align="center">5000</td>
            <td align="center"><a href="https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_nyt10.sh">Link</a></td>
        </tr>
        <tr>
            <td align="center">Re-TACRED</td>
            <td align="center">News</td>
            <td align="center">40</td>
            <td align="center">58465</td>
            <td align="center">19584</td>
            <td align="center">13418</td>
            <td align="center"><a href="https://github.com/gstoica27/Re-TACRED">Link</a></td>
        </tr>
        <tr>
            <td align="center">SciERC</td>
            <td align="center">Scientific</td>
            <td align="center">7</td>
            <td align="center">1366</td>
            <td align="center">187</td>
            <td align="center">397</td>
            <td align="center"><a href="http://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/">Link</a></td>
        </tr>
        <tr>
            <td align="center">SemEval2010</td>
            <td align="center">General</td>
            <td align="center">19</td>
            <td align="center">6507</td>
            <td align="center">1493</td>
            <td align="center">2717</td>
            <td align="center"><a href="https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_semeval.sh">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACRED</td>
            <td align="center">News</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://nlp.stanford.edu/projects/tacred/">Link</a></td>
        </tr>
        <tr>
            <td align="center">TACREV</td>
            <td align="center">News</td>
            <td align="center">42</td>
            <td align="center">68124</td>
            <td align="center">22631</td>
            <td align="center">15509</td>
            <td align="center"><a href="https://github.com/DFKI-NLP/tacrev">Link</a></td>
        </tr>
        <tr>
            <td align="center" rowspan="7"><strong>EE</strong></td>
            <td align="center">ACE05</td>
            <td align="center">News</td>
            <td align="center">33/22</td>
            <td align="center">17172</td>
            <td align="center">923</td>
            <td align="center">832</td>
            <td align="center"><a href="https://catalog.ldc.upenn.edu/LDC2006T06">Link</a></td>
        </tr>
        <tr>
            <td align="center">CASIE</td>
            <td align="center">Cybersecurity</td>
            <td align="center">5/26</td>
            <td align="center">11189</td>
            <td align="center">1778</td>
            <td align="center">3208</td>
            <td align="center"><a href="https://github.com/Ebiquity/CASIE">Link</a></td>
        </tr>
        <tr>
            <td align="center">GENIA11</td>
            <td align="center">Biomedical</td>
            <td align="center">9/11</td>
            <td align="center">8730</td>
            <td align="center">1091</td>
            <td align="center">1092</td>
            <td align="center"><a href="https://bionlp-st.dbcls.jp/GE/2011/eval-test/">Link</a></td>
        </tr>
        <tr>
            <td align="center">GENIA13</td>
            <td align="center">Biomedical</td>
            <td align="center">13/7</td>
            <td align="center">4000</td>
            <td align="center">500</td>
            <td align="center">500</td>
            <td align="center"><a href="https://bionlp-st.dbcls.jp/GE/2013/eval-test/">Link</a></td>
        </tr>
        <tr>
            <td align="center">PHEE</td>
            <td align="center">Biomedical</td>
            <td align="center">2/16</td>
            <td align="center">2898</td>
            <td align="center">961</td>
            <td align="center">968</td>
            <td align="center"><a href="https://github.com/ZhaoyueSun/PHEE">Link</a></td>
        </tr>
        <tr>
            <td align="center">RAMS</td>
            <td align="center">News</td>
            <td align="center">139/65</td>
            <td align="center">7329</td>
            <td align="center">924</td>
            <td align="center">871</td>
            <td align="center"><a href="https://nlp.jhu.edu/rams/">Link</a></td>
        </tr>
        <tr>
            <td align="center">WikiEvents</td>
            <td align="center">Wikipedia</td>
            <td align="center">50/59</td>
            <td align="center">5262</td>
            <td align="center">378</td>
            <td align="center">492</td>
            <td align="center"><a href="https://github.com/raspberryice/gen-arg">Link</a></td>
        </tr>
    </tbody>
</table>
