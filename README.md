
# Awesome-LLM4IE-Papers

\[Will update soon !!\]


Awesome papers about generative Information extraction using LLMs

<p align="center" width="80%">
<img src="./image/intro.png" style="width: 50%">
</p>

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
|  [2INER: Instructive and In-Context Learning on Few-Shot Named Entity Recognition](https://aclanthology.org/2023.findings-emnlp.259/)  |   EMNLP Findings    |  2023-12   |  |
|[In-context Learning for Few-shot Multimodal Named Entity Recognition](https://aclanthology.org/2023.findings-emnlp.196/)  |   EMNLP Findings    |  2023-12   |  |
|  [Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!](https://arxiv.org/abs/2303.08559)  |   EMNLP Findings    |  2023-12   | [Github](https://github.com/mayubo2333/LLM-IE) |
|  [Learning to Rank Context for Named Entity Recognition Using a Synthetic Dataset](https://arxiv.org/abs/2310.10118)  |   EMNLP     |  2023-12   | [Github](https://github.com/CompNet/conivel/tree/gen) |
|  [LLMaAA: Making Large Language Models as Active Annotators](https://arxiv.org/abs/2310.19596)  |   EMNLP Findings    |  2023-12   | [Github](https://github.com/ridiculouz/LLMAAA) |
|  [Prompting ChatGPT in MNER: Enhanced Multimodal Named Entity Recognition with Auxiliary Refined Knowledge](https://arxiv.org/abs/2305.12212)  |   EMNLP Findings    |  2023-12   | [Github](https://github.com/JinYuanLi0012/PGIM) |
|  [Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models](https://arxiv.org/abs/2311.08921)  |   Arxiv    |  2023-11   |  |
|  [GPT-NER: Named Entity Recognition via Large Language Models](https://arxiv.org/abs/2304.10428)  |   Arxiv    |  2023-10   | [Github](https://github.com/ShuheWang1998/GPT-NER) |
|  [Prompt-NER: Zero-shot Named Entity Recognition in Astronomy Literature via Large Language Models](https://arxiv.org/abs/2310.17892)  |   Arxiv    |  2023-10   |  |
|[Inspire the Large Language Model by External Knowledge on BioMedical Named Entity Recognition](https://arxiv.org/abs/2309.12278)  |   Arxiv    |  2023-09   |  |
|[One Model for All Domains: Collaborative Domain-Prefx Tuning for Cross-Domain NER](https://arxiv.org/abs/2301.10410)  |   IJCAI    |  2023-09   | [Github](https://github.com/zjunlp/DeepKE/tree/main/example/ner/cross) |
|  [Chain-of-Thought Prompt Distillation for Multimodal Named Entity Recognition and Multimodal Relation Extraction](https://arxiv.org/abs/2306.14122)  |   Arxiv    |  2023-08   |  |
|[UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition](https://arxiv.org/abs/2308.03279)  |   Arxiv    |  2023-08   | [Github](https://github.com/universal-ner/universal-ner) |
|[Debiasing Generative Named Entity Recognition by Calibrating Sequence Likelihood](https://aclanthology.org/2023.acl-short.98/)  |   ACL Short    |  2023-07   |  |
|  [Entity-to-Text based Data Augmentation for various Named Entity Recognition Tasks](https://aclanthology.org/2023.findings-acl.578/)  |   ACL Findings     |  2023-07   |  |
|  [Large Language Models as Instructors: A Study on Multilingual Clinical Entity Extraction](https://aclanthology.org/2023.bionlp-1.15/)  |   BioNLP    |  2023-07   | [Github](https://github.com/arkhn/bio-nlp2023) |
|  [PromptNER : Prompting For Named Entity Recognition](https://arxiv.org/abs/2305.15444)  |   Arxiv    |  2023-06   | [Github](https://github.com/tricktreat/PromptNER) |
|  [Does Synthetic Data Generation of LLMs Help Clinical Text Mining?](https://arxiv.org/abs/2303.04360)  |   Arxiv    |  2023-04   |  |
|  [Structured information extraction from complex scientific text with fine-tuned large language models](https://arxiv.org/abs/2212.05238)  |   Arxiv    |  2022-12   | [Github](http://www.matscholar.com/info-extraction) |
|  [De-bias for generative extraction in unified NER task](https://aclanthology.org/2022.acl-long.59.pdf)  |   ACL      |  2022-05   |  |
|  [Document-level Entity-based Extraction as Template Generation](https://aclanthology.org/2021.emnlp-main.426/)  |   EMNLP      |  2021-11   | [Github](https://github.com/PlusLabNLP/TempGen) |
|  [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223)  |   ACL      |  2021-08   | [Github](https://github.com/yhcc/BARTNER) |
|  [Template-Based Named Entity Recognition Using BART](https://aclanthology.org/2021.findings-acl.161.pdf)  |   ACL Findings    |  2021-08   | [Github](https://github.com/Nealcly/templateNER) |


## Relation Extraction 
Models targeting only RE tasks.

### Relation Classification
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |
### Relation Triplet
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |

### Relation Strict
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |

## Event Extraction 
Models targeting only EE tasks.

### Event Detection
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |

### Event Argument Extraction
|  Paper  |      Venue    |   Date  | Code |
| :-----: | :--------------: | :------- | :---------: |
|    |          |     |  |
### Event Detection & Argument Extraction
|  Paper  |      Venue    |   Date  | Code |
| :-----: | :--------------: | :------- | :---------: |
|    |          |     |  |


## Universal Information Extraction
Unified models targeting multiple IE tasks.
### NL-LLMs based
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |
### Code-LLMs based
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |


# Learning Paradigms
A taxonomy by Learning Paradigms.

## Supervised Fine-tuning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |

## Few-shot
### Few-shot Fine-tuning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |
### In-Context Learning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |

## Zero-shot
### Zero-shot Prompting
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |
### Cross-Domain Learning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |
### Cross-Type Learning
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |

## Data Augmentation
### Data Annotation
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |
### Knowledge Retrieval 
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |
### Inverse Generation
|  Paper  |      Venue    |   Date  | Code |
| :----- | :--------------: | :------- | :---------: |
|    |          |     |  |


# Specific Domain
|  Paper  |  Domain |   Venue    |   Date  | Code |
| :----- | :--------------: | :-------: | :---------: |:---------: |
|  [Prompting ChatGPT in MNER: Enhanced Multimodal Named Entity Recognition <br> with Auxiliary Refined Knowledge](https://arxiv.org/abs/2305.12212)  |   Multimodal      | ENMLP Findings  |  2023-12  | [Github](https://github.com/JinYuanLi0012/PGIM) |
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
|  [Zero-shot Clinical Entity Recognition using ChatGPT](https://arxiv.org/abs/2303.16416)  |      Arxiv   |  2023-03    | |
|  [Thinking about GPT-3 In-Context Learning for Biomedical IE? Think Again](https://aclanthology.org/2022.findings-emnlp.329/)  |   EMNLP Findings   |   2022-12  | [Github](https://github.com/dki-lab/few-shot-bioIE) |
|  [Large Language Models are Few-Shot Clinical Information Extractors](https://arxiv.org/abs/2205.12689)  |   EMNLP   |  2022-12   | [Huggingface](https://huggingface.co/datasets/mitclinicalml/clinical-ie) |
