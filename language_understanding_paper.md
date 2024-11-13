
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



Step-by-Step Explanation of Pre-training and Fine-tuning ​
1. Unsupervised Pre-training

Objective: Learn a high-capacity language model from a large corpus of text. ​
Data: Use an unsupervised corpus of tokens (e.g., BooksCorpus). ​
Process:

Language Modeling Objective: Maximize the likelihood of predicting a token given its preceding context. ​
Model Architecture: Use a multi-layer Transformer decoder. ​
Training: Apply multi-headed self-attention and position-wise feedforward layers to produce an output distribution over target tokens. ​
Optimization: Train the model parameters using stochastic gradient descent. ​



2. Supervised Fine-tuning

Objective: Adapt the pre-trained model to specific discriminative tasks using labeled data. ​
Data: Use a labeled dataset where each instance consists of input tokens and a corresponding label. ​
Process:

Input Processing: Pass the input tokens through the pre-trained model to obtain the final transformer's block activation. ​
Output Layer: Feed the activation into an added linear output layer to predict the label. ​
Objective: Maximize the likelihood of the correct label given the input tokens. ​
Auxiliary Objective: Optionally include language modeling as an auxiliary objective to improve generalization and accelerate convergence. ​



3. Task-specific Input Transformations ​

Textual Entailment: Concatenate premise and hypothesis with a delimiter token. ​
Similarity Tasks: Process both possible sentence orderings independently and combine their representations.
Question Answering and Commonsense Reasoning: Concatenate document context, question, and each possible answer with delimiters, process independently, and normalize via a softmax layer. ​

4. Evaluation

Tasks: Evaluate on various tasks such as natural language inference, question answering, semantic similarity, and text classification. ​
Metrics: Use accuracy, F1 score, Pearson correlation, etc., depending on the task. ​

This approach leverages the strengths of unsupervised learning to capture linguistic information and fine-tunes the model to perform well on specific tasks with minimal architectural changes. ​

The role of the Transformer in pre-training is crucial for several reasons:


Capturing Long-Range Dependencies:

The Transformer architecture, with its self-attention mechanism, is particularly effective at capturing long-range dependencies in text. ​ This is essential for understanding context over long sequences, which is a limitation in models like LSTMs. ​



Structured Memory:

The Transformer provides a more structured memory for handling dependencies in text. ​ This structured memory helps in better capturing the relationships between different parts of the text, which is beneficial for various downstream tasks. ​



Scalability:

The Transformer can be scaled up to handle large datasets and complex tasks. Its architecture allows for parallel processing, making it more efficient for training on large corpora.



Versatility:

The Transformer is versatile and can be adapted to various tasks with minimal changes. ​ This makes it an ideal choice for a pre-trained model that needs to be fine-tuned for different specific tasks. ​



Multi-Headed Self-Attention:

The multi-headed self-attention mechanism allows the model to focus on different parts of the input sequence simultaneously. ​ This helps in capturing various aspects of the text, such as syntax and semantics, more effectively. ​



Position-Wise Feedforward Layers:

These layers help in transforming the input representations into more useful features for the language modeling task. They contribute to the model's ability to generate coherent and contextually relevant text. ​



Overall, the Transformer's architecture and mechanisms make it highly effective for pre-training a language model, enabling it to learn rich representations from large amounts of unlabeled text. ​ These representations can then be fine-tuned for specific tasks, leading to significant performance improvements. ​



The architecture of the Transformer used in this study is based on the original Transformer model but is specifically adapted for language modeling. ​ Here are the key components and features of the architecture:
1. Multi-Layer Transformer Decoder ​

Layers: The model consists of 12 layers.
Self-Attention Heads: Each layer has multiple self-attention heads (12 heads in this study). ​
Dimensionality: The model uses 768-dimensional states for the self-attention heads. ​

2. Self-Attention Mechanism

Multi-Headed Self-Attention: This mechanism allows the model to focus on different parts of the input sequence simultaneously, capturing various aspects of the text. ​
Masked Self-Attention: In the context of language modeling, the self-attention mechanism is masked to ensure that the prediction for a token only depends on the previous tokens and not on future tokens. ​

3. Position-Wise Feedforward Layers ​

Feedforward Networks: Each layer includes position-wise feedforward networks with 3072-dimensional inner states. ​
Activation Function: The Gaussian Error Linear Unit (GELU) is used as the activation function. ​

4. Embeddings

Token Embeddings: The model uses a token embedding matrix to convert input tokens into dense vectors. ​
Position Embeddings: Learned position embeddings are used instead of the sinusoidal version to encode the position of each token in the sequence. ​

5. Regularization

Dropout: Dropout is applied to the residual, embedding, and attention layers with a rate of 0.1. ​
L2 Regularization: A modified version of L2 regularization is applied to all non-bias or gain weights. ​

6. Optimization

Adam Optimizer: The Adam optimization scheme is used with a maximum learning rate of 2.5e-4. ​
Learning Rate Schedule: The learning rate is increased linearly from zero over the first 2000 updates and then annealed to zero using a cosine schedule. ​

7. Training Details

Batch Size: The model is trained on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens. ​
Epochs: The model is trained for 100 epochs. ​

8. Byte-Pair Encoding (BPE) ​

Vocabulary: A BPE vocabulary with 40,000 merges is used to handle rare words and subword units. ​

9. Layer Normalization

LayerNorm: Layer normalization is used extensively throughout the model to stabilize and accelerate training. ​

10. Pre-training Objective

Language Modeling: The model is trained to predict the next token in a sequence, given the previous tokens, using a standard language modeling objective. ​

Summary
The Transformer architecture used in this study is a 12-layer decoder-only model with multi-headed self-attention and position-wise feedforward layers. ​ It employs various regularization techniques and optimization strategies to effectively learn from large corpora of text. This architecture is well-suited for capturing long-range dependencies and structured memory, making it highly effective for pre-training a language model that can be fine-tuned for various downstream tasks. ​

The role of unsupervised pre-training in this study is pivotal for several reasons:
1. Learning Universal Representations ​

Objective: The primary goal of unsupervised pre-training is to learn a universal representation of language that can be effectively transferred to various downstream tasks. ​
Method: This is achieved by training a language model on a large corpus of unlabeled text, allowing the model to capture a wide range of linguistic patterns and structures. ​

2. Leveraging Large Unlabeled Corpora ​

Data Availability: Unlabeled text data is abundant and easily accessible, unlike labeled data which is often scarce and expensive to obtain. ​
Utilization: By leveraging these large unlabeled corpora, the model can learn from a vast amount of data, improving its ability to generalize to different tasks. ​

3. Improving Performance on Downstream Tasks ​

Transfer Learning: The representations learned during pre-training can be fine-tuned with minimal labeled data for specific tasks, leading to significant performance improvements. ​
Task-Agnostic Model: The pre-trained model is designed to be task-agnostic, meaning it can be adapted to a wide range of tasks with minimal changes to its architecture. ​

4. Handling Long-Range Dependencies ​

Contextual Understanding: The pre-training process helps the model understand long-range dependencies in text, which is crucial for tasks that require context over long sequences. ​
Structured Memory: The Transformer architecture used in pre-training provides a structured memory that enhances the model's ability to capture and utilize long-term dependencies. ​

5. Reducing the Need for Extensive Supervision ​

Efficiency: Unsupervised pre-training reduces the reliance on large amounts of labeled data, making it more efficient and cost-effective.
Generalization: Models pre-trained in an unsupervised manner tend to generalize better to new tasks and domains, as they have been exposed to a diverse range of linguistic phenomena. ​

6. Auxiliary Objectives

Auxiliary Language Modeling: Including language modeling as an auxiliary objective during fine-tuning helps improve the generalization of the supervised model and accelerates convergence. ​

Summary
Unsupervised pre-training serves as a foundational step in the training process, enabling the model to learn rich, universal representations from large amounts of unlabeled text. ​ These representations can then be fine-tuned for specific tasks, leading to improved performance and generalization. ​ This approach leverages the abundance of unlabeled data, reduces the need for extensive labeled data, and enhances the model's ability to handle long-range dependencies and diverse linguistic patterns.


The purpose of supervised fine-tuning in this study is to adapt the pre-trained language model to specific downstream tasks using labeled data. ​ Here are the key aspects and purposes of supervised fine-tuning:
1. Task-Specific Adaptation

Objective: Fine-tuning adjusts the pre-trained model's parameters to optimize performance on a particular task, such as question answering, text classification, or semantic similarity. ​
Method: This is done by training the model on a labeled dataset specific to the target task, allowing it to learn task-specific patterns and nuances. ​

2. Leveraging Pre-Trained Representations

Transfer Learning: The representations learned during the unsupervised pre-training phase are leveraged and refined during fine-tuning. ​ This helps the model quickly adapt to the new task with improved performance. ​
Efficiency: Fine-tuning requires significantly less labeled data and computational resources compared to training a model from scratch, as the model already has a strong foundation from pre-training.

3. Minimal Architectural Changes ​

Consistency: The study emphasizes using minimal changes to the model architecture during fine-tuning. ​ This approach maintains the benefits of the pre-trained model while adapting it to new tasks. ​
Task-Specific Input Transformations: For tasks with structured inputs (e.g., question answering or textual entailment), the study uses input transformations to convert these inputs into a format compatible with the pre-trained model, avoiding extensive architectural modifications. ​

4. Improving Generalization and Performance ​

Generalization: Fine-tuning helps the model generalize better to the specific task by learning from labeled examples, which provide clear guidance on the task's requirements. ​
Performance Gains: The study demonstrates that fine-tuning the pre-trained model leads to significant performance improvements across various benchmarks, often surpassing state-of-the-art results. ​

5. Auxiliary Objectives

Auxiliary Language Modeling: Including language modeling as an auxiliary objective during fine-tuning can further improve the model's generalization and accelerate convergence, as it continues to benefit from the unsupervised learning signals. ​

Summary
Supervised fine-tuning serves to adapt the pre-trained language model to specific downstream tasks by training it on labeled data. ​ This process refines the model's parameters to optimize task-specific performance, leveraging the rich representations learned during pre-training. ​ Fine-tuning is efficient, requires minimal architectural changes, and significantly enhances the model's ability to generalize and perform well on a variety of tasks. ​





