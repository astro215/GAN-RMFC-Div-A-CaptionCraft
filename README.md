Github - https://github.com/astro215/GAN-RMFC-Div-A-CaptionCraft/edit/main


cgt-gans
--

- github - https://github.com/astro215/cgt-gans/blob/main/training/cgt-gans-test-8.ipynb
- kaggle - https://www.kaggle.com/code/astro215/cgt-gans-test



datasets
- 
SciCap 
- graphs dataset (SciCap)- https://github.com/tingyaohsu/SciCap
- custom split -
- hugging-face - https://huggingface.co/datasets/astro21/private_gans_split	
- kaggel - https://www.kaggle.com/datasets/jainilpatelbtech2021/gans-dataset-cp/versions/1
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
			configs:
			- config_name: default
			  data_files:
			  - split: train
			    path: data/train-*
			  - split: val
			    path: data/val-*
			  - split: test
			    path: data/test-*
			    
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
	- 
		

	- **Coco2014** - Can't identify the objects correctly.


**Cgt-GANs**

- **Datasets** - Coco2014
- **Notebooks** - https://github.com/astro215/cgt-gans/tree/main/training
- ** Deployment** - 








# References
- WGANS - https://github.com/bastienvanderplaetse/gan-image-captioning 
- Pix2Struct - https://arxiv.org/abs/2210.03347 , 
- 
- 
