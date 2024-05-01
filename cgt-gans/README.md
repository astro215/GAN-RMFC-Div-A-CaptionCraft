cgt-gans
--

- github - https://github.com/astro215/cgt-gans/blob/main/training/cgt-gans-test-8.ipynb
- kaggle - https://www.kaggle.com/code/astro215/cgt-gans-test


# **Cgt-GANs**
- The model involves training a generative model (generator) and a discriminative model (discriminator) using a paired dataset of image and caption embeddings. The objective is to generate captions that are contextually and semantically aligned with given images. The system uses a GAN-like architecture where the generator tries to create plausible captions, and the discriminator evaluates them.


### Generator Model Architecture
The generator model is based on the `ClipCaptionModel`, which leverages a GPT-2 structure modified to interact with image features:

- **GPT2LMHeadModel**: The core is a standard GPT-2 model adapted for language modeling and text generation. This part consists of:
  - **Embedding Layers**: Word and position embeddings encode the input tokens and their positions within a sequence.
  - **Transformer Blocks**: Comprised of multiple layers, each containing a self-attention mechanism and a feed-forward network (MLP).
  - **LayerNorm and Dropout**: Used across different points in the transformer to normalize activations and prevent overfitting.

- **Clip Projector (MLP)**: This Multi-Layer Perceptron takes embeddings from a CLIP model (trained separately to encode image data) and projects them into the space where GPT-2 embeddings reside. This fusion enables the model to generate text based on visual input.

### Discriminator Model Architecture
The discriminator is structured around a `RobertaDiscriminator`:

- **RoBERTa Model**: An adaptation of the BERT architecture optimized for more robust performance in language understanding tasks.
  - **RobertaEmbeddings**: Handles embeddings related to words, positions, and token types.
  - **RobertaEncoder**: Contains multiple encoding layers that process text through self-attention and feed-forward networks.
  - **RobertaPooler**: Processes the output of the last encoding layer to produce a fixed-size representation.

- **MLP (Discriminator Head)**: A simple MLP that processes the output of the RoBERTa model to determine the authenticity of the generated captions (real vs. generated).

### Losses
- **BCEWithLogitsLoss**: Used by the discriminator to classify whether the input captions are real or fake, facilitating the training of the GAN to improve the realism of the generated captions.

### Optimization
- **Adam Optimizer**: Typically used for training both the generator and discriminator, facilitating efficient stochastic optimization with adaptive estimation of lower-order moments.

### Interesting Insights
- **Integration of CLIP and GPT**: The merging of visual and textual models (CLIP for visual embeddings and GPT-2 for text generation) is a notable innovation that enhances the generator's ability to create relevant and contextually appropriate captions based on visual inputs.
- **Training Requirements**: As noted, the uninitialized weights (e.g., RoBERTa's pooler layer) indicate the need for further fine-tuning and training on a downstream task specific to the application, underscoring the adaptability and potential customization of the model.


 **Model Architecture**

- **Generator**
The generator is responsible for creating captions based on image embeddings. It utilizes a GPT-2 model to generate text and a transformer or MLP (Multi-Layer Perceptron) for mapping image embeddings to the GPT-2 input space. The generator takes a prefix embedding from an image and transforms it into a sequence of embeddings that serve as a conditioned prefix for the GPT-2 model. The GPT-2 model then generates a caption based on this prefix.

- **Discriminator**
The discriminator evaluates the plausibility of the generated captions relative to real captions and images. It uses a Roberta model that encodes the textual input (either generated or real captions) and then passes this encoding through an MLP to compute a score indicating the realism of the text.

Loss Functions

- **Generator Loss (G_loss)**
The generator loss is computed as a combination of the reinforcement learning reward (`reward_loss`) and the feature discriminator loss (`fd_loss`):



	$$
	G\_loss = \{reward\_weight} \times \{reward\_loss} + (1 - \{reward\_weight}) \times \{fd\_loss}
	$$



- **reward_loss**: This is a policy gradient loss used to optimize the generator in a reinforcement learning setup where the generator's output (caption) is treated as an action. The loss encourages the generator to produce actions that lead to higher rewards.

- **fd_loss**: This loss is computed as a function of how well the generator’s outputs (captions) can fool the discriminator into thinking that they are real, effectively using the discriminator as a critic.

### Discriminator Loss (D_loss)
The discriminator loss is a binary cross-entropy loss calculated over real and generated captions. The goal is to correctly classify real captions as real and generated captions as fake.

$$
D\_loss = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(p(y_i)) + (1 - y_i) \cdot \log(1 - p(y_i)) \right]
$$

where $$\( p(y_i) \)$$ is the discriminator's probability estimate for the i-th example being real, and $$\( y_i \)$$ is the true label (1 for real, 0 for generated).

**Reward System**

The reward system is central to training the generator. It involves calculating a reward for each generated caption based on various metrics:

- **Clip score**: Measures the semantic similarity between the generated caption and the image.
- **Cosine score and L1 score**: These scores measure the similarity between the generated caption and the real captions associated with the image.

The rewards are computed by evaluating the generated captions using a discriminator, which scores the captions based on their plausibility. These scores are used to compute the `reward_loss` in the generator loss function.

- **Example of Reward Mechanism**

	Suppose the generator produces two captions:
	1. "A dog playing in the park."
	2. "An animal is outside."

	Assuming the discriminator scores them as 0.8 and 0.6, respectively, and the baseline (greedy output) score is 0.5. The rewards would be:
	- For caption 1: $$\( 0.8 - 0.5 = 0.3 \)$$
	- For caption 2: $$\( 0.6 - 0.5 = 0.1 \)$$

	The generator will then use these rewards to adjust its parameters to increase the probability of generating captions that receive higher rewards.


### 1. **ClipCaptionModel**

In `ClipCaptionModel`, the loss used is primarily derived from the model's forward operation during training, which involves the generation of text from image features. However, the actual training details are somewhat obscured as they depend on how the forward pass is utilized in training loops elsewhere in the implementation. From the implementation provided:

- **Log-Likelihood Loss:** This model utilizes the GPT-2 architecture for generating text. During training, if labels (i.e., expected correct outputs) are provided, the GPT-2 model computes the cross-entropy loss between predicted logits and actual token labels. This can be understood as minimizing the negative log likelihood of the correct class, which is a common loss in training language models.
  
  Mathematically:
  $$\[
  \text{Loss} = -\sum_{i=1}^{N} \log(p_{yi})
  \]$$
  where $$ p_{yi} $$ is the probability assigned by the model to the true class $$ y_i $$ for the $$ i^{th} $$ instance.

### 2. **RobertaDiscriminator**

In `RobertaDiscriminator`, the goal is to discern real from fake captions relative to the input images. The main components here are:

- **Binary Cross-Entropy Loss (BCELoss):** The model predicts a single scalar (logit) for each text input. This scalar is passed through a sigmoid function to estimate the probability that the input text is a genuine caption for the associated image. BCELoss measures the difference between the predicted probability and the actual label (real or fake).
  
  Mathematically, for binary classification:
  $$
  {BCELoss}= -\left(y \cdot \log(\sigma(x)) + (1 - y) \cdot \log(1 - \sigma(x))\right)
  $$
  where $${y}$$ is the true label, $$ x $$is the model logit, and $$ \sigma$$ is the sigmoid function converting logits to probabilities.

### 3. **Scoring Mechanisms**

- **ClipDiscriminator Scoring:**
  The `ClipDiscriminator` computes several metrics to evaluate the quality of generated captions relative to the true image content. These include L1 loss, cosine similarity, and a proprietary CLIP score, reflecting how well the captions are aligned semantically with the image features:
  
  - **L1 Loss:** Measures the mean absolute difference between ground-truth image features and features generated by the CLIP model from the captions.
    $$
    \text{L1 Loss} = \frac{1}{n} \sum |x_{\text{true}} - x_{\text{pred}}|
    $$
  
  - **Cosine Similarity:** Measures the cosine of the angle between the normalized feature vectors of the ground-truth and generated captions. Higher similarity indicates better alignment.
   $$
    \text{Cosine Similarity} = \frac{x \cdot y}{\|x\|\|y\|}
    $$
    where $$ x$$ and $$y $$ are the feature vectors of the true and predicted captions, respectively.

  - **CLIP Score:** A proprietary metric from OpenAI's CLIP model, which leverages the ability of CLIP to assess similarity between text and images effectively. It is essentially a scaled dot product of the normalized feature vectors.
    $$
    \text{CLIP Score} = e^{s} \cdot \left(\frac{\text{features}_{\text{image}} \cdot \text{features}_{\text{text}}^T}{\|\text{features}_{\text{image}}\| \|\text{features}_{\text{text}}\|}\right)
   $$
    where $$ s $$ is the learned scale parameter in the CLIP model.


**Conclusion**

The model is a sophisticated system that integrates generative and discriminative approaches to produce and evaluate text based on image data. Through iterative training, involving both the generator and discriminator, the system learns to generate captions that are not only plausible but also contextually relevant to the images. The use of various loss functions and a reward system plays a crucial role in refining the model's performance, ensuring that the captions are both diverse and accurate representations of the image content.


- **Datasets** - Coco2014
- **Notebooks** - https://github.com/astro215/cgt-gans/tree/main/training
- **Deployment** - 








# References 
- https://github.com/fkodom/clip-text-decoder
- https://medium.com/@uppalamukesh/clipcap-clip-prefix-for-image-captioning-3970c73573bc
- https://github.com/jmisilo/clip-gpt-captioning
- https://arxiv.org/abs/2211.00575
