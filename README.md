**CaptionCraft**
--

- github - https://github.com/astro215/GAN-RMFC-Div-A-CaptionCraft/blob/main/caption-craft-gans/training/caption-craft-gans-train-8.ipynb
- kaggle - https://www.kaggle.com/code/astro215/caption-craft-gans-test







https://github.com/astro215/GAN-RMFC-Div-A-CaptionCraft/assets/111174198/a0c35328-b831-4f98-a108-917b7903694d

- Deployment - [Gradio](https://huggingface.co/spaces/astroCodes/caption-craft-gans)



**Datasets**
- 
SciCap 
- graphs dataset (SciCap)- https://github.com/tingyaohsu/SciCap
- custom split
	- hugging-face - https://huggingface.co/datasets/astro21/private_gans_split/blob/main/README.md	
	- kaggle - https://www.kaggle.com/datasets/jainilpatelbtech2021/gans-dataset-cp/versions/1
		- metadata  

			  features:
			  - name: image
			    dtype: image
			  - name: folder
			    dtype: string
			  - name: caption
			    dtype: string
			  splits:
			  - name: train
			    num_bytes: 3188186445.4861555
			    num_examples: 106834
			  - name: val
			    num_bytes: 407861081.1096169
			    num_examples: 13354
			  - name: test
			    num_bytes: 389676044.3902272
			    num_examples: 13355
			  download_size: 4074942870
			  dataset_size: 3985723570.9859996
			
			    
- pre-processed (.npy , .txt features , captions ) - https://www.kaggle.com/datasets/jainilpatelbtech2021/dataset-gans-train

			
_____________
Coco2014
- images-(coco dataset) - https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3/code?datasetId=1573501&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false

- pre-processed - https://drive.google.com/drive/folders/1v0LjImTb3whgPuh7RVDIAk20j2_ai49p?usp=sharing



Models
------------------------------------------------------ 
**WGANs**  


1. **Architecture and Components**: The **Generator** synthesizes new sequences using embeddings, GRU cells, and attention mechanisms, while the **Discriminator** evaluates sequences for authenticity using a similar setup.
2. **Wasserstein Loss and Training Dynamics**: Utilizes Wasserstein loss to stabilize training and address traditional GAN issues by calculating based on discriminator scores.
3. **Gradient Penalty**: Implements a gradient penalty to enforce the Lipschitz constraint on the discriminator, crucial for maintaining training stability and effectiveness.
4. **Optimization and Regularization Techniques**: Features dropout and embedding weight management, with training involving alternating discriminator and generator updates to maintain balance.


- **Datasets** - SciCap , Coco2014
- **Notebooks** - 
	- **SciCap** - https://www.kaggle.com/code/jainilpatelbtech2021/wgan-test1/notebook
	- **Coco2014** -https://www.kaggle.com/code/jainilpatelbtech2021/wgan-f

- **Results**
	- **SciCap** - Repeating one word in whole sentence for each image
		<a href="https://ibb.co/98TKh3z"><img src="https://i.ibb.co/Tv8j1qp/Screenshot-2024-04-30-210845.png" alt="Screenshot-2024-04-30-210845" border="0"></a>

	- **Coco2014** - Can't identify the objects correctly.

		![Coco2014](https://i.ibb.co/cyNgPHf/top4.png)


**Pix2Struc**

1. **Architecture and Components:**
   - **Encoder-Decoder Framework:** Pix2Struct utilizes a sophisticated encoder-decoder structure. The encoder is designed for visual inputs with patch projection converting images into a sequence of embeddings, while the decoder focuses on text generation.
   - **Attention Mechanisms:** The model features specialized vision and text attention mechanisms that facilitate effective cross-modal understanding and integration, making it adept at tasks requiring the transformation of visual inputs into textual outputs.

2. **Losses and Training:**
   - **Pretraining on Web Data:** Pix2Struct is pretrained by parsing masked screenshots of web pages into simplified HTML. This method leverages the natural alignment between visual elements and their HTML descriptors to teach the model robust visual-textual associations.
   - **Comprehensive Pretraining Objective:** The model's pretraining encompasses learning signals typical of OCR, language modeling, and image captioning, providing a multifaceted foundation for downstream tasks.

3. **Optimization:**
   - **Variable-Resolution Input:** The model can process inputs at various resolutions, allowing it to adapt to different image qualities and sizes seamlessly.
   - **Fine-Tuning:** For specific tasks such as image captioning, Pix2Struct is further optimized by fine-tuning on task-specific datasets, ensuring the model's performance is tailored to the unique characteristics of the target application.

4. **Integration of Language and Vision:**
   - **Language Prompts in Visual Contexts:** One of Pix2Struct’s standout features is its ability to integrate language prompts directly with visual inputs. This capability is crucial for tasks like visual question answering, where the model must interpret and respond to textual queries in light of the visual data presented.
   - **Cross-Modal Attention:** This feature enables the model to attend specifically to relevant areas within the image when generating text, ensuring that the textual output is contextually aligned with the visual input.

- **Datasets** - SciCap 
- **Notebooks** 
	- **SciCap** - https://www.kaggle.com/code/astronlp/caption-pretrained

- **Results**
	- **SciCap** - Just making captions around the OCR text extracted from the patches of image.


**caption-craft-gans**
- The model involves training a generative model (generator) and a discriminative model (discriminator) using a paired dataset of image and caption embeddings. The objective is to generate captions that are contextually and semantically aligned with given images. The system uses a GAN-like architecture where the generator tries to create plausible captions, and the discriminator evaluates them.

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
[ D\_loss = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(p(y_i)) + (1 - y_i) \cdot \log(1 - p(y_i)) \right]]
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

**Conclusion**

The model is a sophisticated system that integrates generative and discriminative approaches to produce and evaluate text based on image data. Through iterative training, involving both the generator and discriminator, the system learns to generate captions that are not only plausible but also contextually relevant to the images. The use of various loss functions and a reward system plays a crucial role in refining the model's performance, ensuring that the captions are both diverse and accurate representations of the image content.


- **Datasets** - Coco2014
- **Notebooks** - https://github.com/astro215/GAN-RMFC-Div-A-CaptionCraft/blob/main/caption-craft-gans/training



# References
- WGANS - https://github.com/bastienvanderplaetse/gan-image-captioning 
- Pix2Struct - https://arxiv.org/abs/2210.03347 , 
- Clip
	- https://github.com/fkodom/clip-text-decoder
	- https://medium.com/@uppalamukesh/clipcap-clip-prefix-for-image-captioning-3970c73573bc
	- https://github.com/jmisilo/clip-gpt-captioning
	- https://arxiv.org/abs/2211.00575


# Literature Survey 
- TransGAN - https://arxiv.org/abs/2102.07074
- Attention Is All You Need - https://arxiv.org/abs/1706.03762
- GANsformer - https://arxiv.org/pdf/2103.01209v2
- ViTGAN - https://arxiv.org/pdf/2107.04589
- WGANS - https://github.com/bastienvanderplaetse/gan-image-captioning 
- Pix2Struct - https://arxiv.org/abs/2210.03347 , 
- Clip
	- https://github.com/fkodom/clip-text-decoder
	- https://medium.com/@uppalamukesh/clipcap-clip-prefix-for-image-captioning-3970c73573bc
	- https://github.com/jmisilo/clip-gpt-captioning
	- https://arxiv.org/abs/2211.00575

