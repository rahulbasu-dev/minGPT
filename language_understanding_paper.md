
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf


The document discusses a semi-supervised approach for natural language understanding tasks using generative pre-training followed by discriminative fine-tuning, achieving significant improvements in various language understanding benchmarks.
Improving Language Understanding by Generative Pre-Training
Contact details for Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever at OpenAI.
Abstract
Generative pre-training on unlabeled text followed by task-specific fine-tuning significantly improves performance, achieving gains of 8.9% on commonsense reasoning, 5.7% on question answering, and 1.5% on textual entailment.
1 Introduction
The paper explores a semi-supervised approach using unsupervised pre-training and supervised fine-tuning for various language understanding tasks.

The paper addresses the challenge of learning from raw text to reduce dependence on supervised learning in NLP.
It proposes a semi-supervised approach combining unsupervised pre-training with supervised fine-tuning to learn universal representations.
The approach uses a Transformer model, pre-trained on a large corpus of unlabeled text, and fine-tuned on specific tasks.
Task-specific input adaptations enable effective fine-tuning with minimal changes to the pre-trained model architecture.
The model outperforms task-specific architectures, achieving significant improvements in 9 out of 12 tasks, including 8.9% on commonsense reasoning and 5.7% on question answering.
Evaluations on natural language inference, question answering, semantic similarity, and text classification show robust transfer performance.
Zero-shot analysis demonstrates the model's acquisition of useful linguistic knowledge for downstream tasks.

2 Related Work
The section discusses semi-supervised learning in NLP, focusing on word embeddings, unsupervised pre-training, and auxiliary training objectives.

The text explores semi-supervised learning for NLP, highlighting its applications in sequence labeling and text classification.
Early methods utilized unlabeled data for word-level statistics, while recent approaches use word embeddings to enhance task performance.
Unsupervised pre-training, a subset of semi-supervised learning, is used to initialize models, improving generalization in deep neural networks.
Pre-training with language modeling objectives followed by fine-tuning on target tasks has shown improvements in text classification, especially with transformer networks.
Auxiliary training objectives, such as language modeling, have been used to enhance performance on various NLP tasks, demonstrating the benefits of unsupervised pre-training.
The text emphasizes minimal changes to model architecture during transfer and the effectiveness of the discussed methods on tasks like natural language inference and paraphrase detection.

3 Framework
The section describes a two-stage training procedure for language models, including pre-training on large text corpora and fine-tuning for specific tasks.

The training procedure consists of two stages: pre-training a high-capacity language model on a large corpus of text, followed by fine-tuning on a discriminative task with labeled data.
The pre-training stage uses a standard language modeling objective to maximize the likelihood of token sequences, modeled with a multi-layer Transformer decoder.
The fine-tuning stage adapts the pre-trained model to a supervised target task using labeled datasets, with an added linear output layer to predict labels.
Including language modeling as an auxiliary objective during fine-tuning improves generalization and accelerates convergence.
Structured inputs for tasks like question answering and textual entailment are transformed into token sequences to be processed by the pre-trained model.
The approach avoids extensive task-specific customization by using a traversal-style method to convert structured inputs into ordered sequences.
The only extra parameters required during fine-tuning are for the linear output layer and embeddings for delimiter tokens.

4 Experiments
The section discusses the performance of a transformer language model on various NLP tasks, achieving state-of-the-art results.

The transformer language model was pre-trained on the BooksCorpus dataset, which contains over 7,000 unique unpublished books.
The model is a 12-layer decoder-only transformer with masked self-attention heads, trained with the Adam optimization scheme.
Fine-tuning was performed on various supervised tasks, including natural language inference, question answering, semantic similarity, and text classification.
The model achieved significant improvements on NLI tasks, with up to 5.8% improvement on QNLI and 5% on SciTail.
In question answering, the model outperformed previous best results by up to 8.9% on Story Cloze and 5.7% on RACE.
For semantic similarity tasks, the model achieved state-of-the-art results on two out of three datasets, with a 1 point gain on STS-B.
The model also showed strong performance in text classification, with a score of 45.4 on CoLA and 91.3% accuracy on SST-2.
Overall, the model achieved new state-of-the-art results in 9 out of 12 datasets, demonstrating its effectiveness across various dataset sizes.

5 Analysis
The section discusses the impact of transferring layers from pre-trained models to target tasks, zero-shot behaviors, and ablation studies.

The section examines the effect of transferring different numbers of layers from unsupervised pre-training to supervised tasks, showing up to a 9% performance improvement on MultiNLI.
Zero-shot performance on various tasks improves steadily with generative pre-training, indicating the effectiveness of the underlying generative model.
Ablation studies reveal that the auxiliary LM objective during fine-tuning benefits larger datasets but not smaller ones.
The Transformer architecture outperforms a single-layer 2048 unit LSTM, with a 5.6 average score drop when using the LSTM.
The LSTM only outperforms the Transformer on the MRPC dataset.
Lack of pre-training results in a 14.8% performance decrease across all tasks compared to the fully pre-trained model.
The section includes a detailed table comparing different model ablations on tasks like CoLA, SS, MRPC, STSB, QQP, MNLI, QNLI, and RTE.

6 Conclusion
A task-agnostic model using generative pre-training and discriminative fine-tuning improved state-of-the-art performance on 9 of 12 datasets, highlighting the effectiveness of Transformers and long-range text dependencies.
