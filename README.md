# GAN-RMFC-Div-A-CaptionCraft


Github - https://github.com/astro215/cgt-gans/tree/main


cgt-gans
--

- github - https://github.com/astro215/cgt-gans/blob/main/training/cgt-gans-test-8.ipynb
- kaggle - https://www.kaggle.com/code/astro215/cgt-gans-test



datasets
- 
SciCap 
- graphs dataset (SciCap)- https://github.com/tingyaohsu/SciCap
- custom split -https://huggingface.co/datasets/astro21/private_gans_split	
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










# References
- WGANS - https://github.com/bastienvanderplaetse/gan-image-captioning 
- Pix2Struct - https://arxiv.org/abs/2210.03347 , 
- 
