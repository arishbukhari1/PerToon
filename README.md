# A Generative AI System for Personalized Cartoon Images and Captions
 
Project By **Paolo Ferrara**|ferrara.pao@northeastern.edu , **Muhammad Arish Salam Bukhari** | bukhari.mu@northeastern.edu, **Shan Lin** | lin.shan1@northeastern.edu, **Patricia Atkinson** | atkinson.p@northeastern.edu 

**May 30th, 2025**

# 1. Project Description 

This project aims to build a Generative AI tool that allows users to create personalized cartoon-style images with expressive captions. A user uploads an image, which is transformed into a cartoon using a fine-tuned image generation model. The user then provides a short prompt or mood, which is used by a fine-tuned GPT model to generate a funny or meaningful caption. The final result is a shareable image that blends vision and language processing. 

The main objective of this project is to deepen our understanding of Generative AI by combining computer vision and natural language processing in one system. We aim to explore how to build a customized, end-to-end AI tool that could be extended for wide usage across different creative applications. 

# Business Value 

We consider this project as a prototype for a Generative AI product targeting the vertical area of digital storytelling. Personalized and expressive content is a growing trend in social media, marketing, and entertainment, and tools that help users generate such content easily are in high demand. Our system architecture could be adapted for meme creation, branded visual content, or even AI-assisted creative writing and art.  

# Innovation 

While both cartoonization and text generation are independently well-studied, this project’s innovation lies in the seamless integration into a unified, expressive pipeline, turning a static photo and brief user prompt into an emotionally resonant, stylized digital product. This approach is not only technically diverse but creatively impactful, serving as a proof-of-concept for learning multimodal Generative AI systems. 

# 2. Problem Statement 

Nowadays, digital storytelling through captioned images has become a common and popular way to express ideas and emotions on digital platforms, thanks to how quickly they can communicate a message and how easily they can be shared. However, the process of creating expressive visual content can be difficult for many people, especially those who lack technical skills or creative experience.  

To address this challenge, we propose a Generative AI system that combines a cartoon-style image model and a text generation model. We want to make digital storytelling fast, easy, and personalized for everyone. In pursuit of this goal, the project seeks to answer the following key questions: 

- How effectively can generative AI personalize visual and textual content based on minimal user input?  
- How can we define and apply metrics to evaluate the quality and effectiveness of the generated content? 
- How do the base models and fine-tuning affect the quality of the output? What are the limitations in aligning generated captions with the user’s intended prompt? 
- Can the combined AI system produce high-quality, engaging outputs that match user expectations? 
- Can this approach be generalized to broader domains like creative arts, education, or marketing? 

# 3. Dataset Selection 

To develop our personalized cartoon image generation system, we require two types of datasets: 

**Visual data** – for training and evaluating cartoon-style image generation models. 
**Textual data** – for training a caption generation model based on user mood or intent. 

We are currently evaluating several publicly available datasets for suitability based on quality, diversity, annotation depth, and licensing. Below, we highlight two representative datasets that align particularly well with our project's goals. 

**Dataset Spotlight:** Cartoon Set 
**Source:** Google AI – Cartoon Set 

Description: A collection of 10k cartoon avatar face images generated with random variation in attributes such as hair color, eye shape, and accessories. The images are 2D vector-style and highly stylized, making them suitable as target data for training image translation models like CartoonGAN. 

**Key Fields:** Image files with associated metadata (e.g., eye shape, skin tone, glasses, etc.) 
Use Case in Our Project: Serves as the target cartoon domain for the cartoonization module, enabling style transfer from real faces to stylized cartoon faces. 
Language/Modality: Visual only (images + structured attributes) 

**Dataset Spotlight:** Memeify 
**Source:** GitHub – Memeify Dataset 

Description: A large-scale collection of meme captions (1.1 million+) across 128 humor or theme categories. Each entry contains a short caption paired with a label indicating its meme category (e.g., sarcasm, politics, work life, relationships). 

**Key Fields:** text (caption), label (meme theme) 
**Use Case in Our Project:** Provides training data for fine-tuning a GPT-2 model to generate expressive, humorous captions conditioned on user-selected moods or themes. 
**Language:** English 

**Alternative Datasets Under Consideration**

To enhance flexibility and expand experimental options during development, we are also considering the following datasets: 

**Selfie2Anime**
Source: https://selfie2anime.com/ 

Description: A paired dataset containing selfie images and their corresponding anime-style transformations, useful for supervised training of image-to-image translation models. 

**CelebA (CelebFaces Attributes Dataset)**

Source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 

Description: Over 200k celebrity face images with 40 attribute labels per image, commonly used in face recognition and generative model training. Useful as a real-world photo domain when paired with cartoon targets. 

**Dataset Evaluation Criteria**

In the prototyping & data Preparation phases of the project, we will evaluate candidate datasets based on: 

**Relevance:** Fit with our goals of cartoon-style transformation and mood-based text generation. 
**Quality:** Resolution, balance, and richness of the data. 
**Annotation Depth:** Presence of facial attributes (visual) or emotion/theme labels (text). 
**Ethical Considerations:** Inclusivity, diversity, and appropriateness of content. 
**Licensing:** Availability for academic research use. 

 

# 4. Background – Base Model and Methodology Analysis 

Generative AI has achieved strong results in both computer vision and natural language processing. For image generation, models such as CartoonGAN and CycleGAN are commonly used to convert real photos into cartoon-style images. These models learn how to apply artistic style while preserving facial structure, making them suitable for applications like avatar creation or creative content generation. CartoonGAN uses adversarial training to stylize facial images while preserving content. CycleGAN enables unpaired image-to-image translation by enforcing cycle-consistency loss, as demonstrated by Zhu et al. (2017). For a clearer overview, Hui (2019) provides a practical explanation of CycleGAN's architecture and training process. 

In the language domain, transformer-based models such as GPT-2 can generate meaningful text when given simple prompts or fine-tuned with labeled examples. Prior work shows that GPT-2 can be adapted to create custom text styles, including emotional or humorous captions (Woolf, 2019). 

Some recent research has explored combining image and text generation. The XMeCap model focuses on aligning image parts with metaphorical text, while the MemeCap dataset includes annotated meme captions for training multimodal systems. Another example, by Gribbly (2022), shows how AI models can generate captions for images using theme-based prompts. 

Our project builds on these ideas by combining cartoon-style image generation with prompt-based captioning to create personalized visual content. 

# References 

Woolf, M. (2019). How To Make Custom AI-Generated Text With GPT-2. Minimaxir. https://minimaxir.com/2019/09/howto-gpt2/ 

Gribbly, A. (2022). Building an AI Meme Generator. Medium. https://medium.com/@gribbly.avax/building-an-ai-meme-generator-c30e1df52343 

Wang, Y. et al. (2024). XMeCap: Cross-modal Meme Captioning. https://arxiv.org/html/2407.17152v2 

Sharma, S. et al. (2023). MemeCap: Understanding Memes with Explanations. EMNLP. https://aclanthology.org/2023.emnlp-main.89.pdf 

Chen, Y. et al. (2018). CartoonGAN: Generative Adversarial Networks for Photo Cartoonization. CVPR. https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf 

Zhu, J. Y. et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV. https://arxiv.org/abs/1703.10593 

Hui, J. (2019). GAN – CycleGAN. Medium. https://jonathan-hui.medium.com/gan-cyclegan-6a50e7600d7 

 

# 5. Methodology 

Our approach follows a structured AI project pipeline that combines computer vision and NLP techniques to generate personalized cartoon-style images with captions. The methodology includes the following main stages: 

**1. Data Collection and Exploratory Data Analysis (EDA)**

We plan to explore and explore separate datasets for the two main components: 

**Cartoonization (Vision):** We are considering facial datasets (e.g., LFW, potentially CelebA) and cartoon-style datasets (e.g., CartoonSet, Selfie2Anime) to support GAN-based stylization. EDA will focus on resolution, facial structure variation, and style consistency to guide model selection and tuning. 

**Caption Generation (Text):** For meme-style captioning, we will curate text data from open datasets such as Memeify and online meme repositories. Initial analysis will examine caption length, language style, and emotional tone diversity. 

**2.  Data Preprocessing**

**Images:** Resize, normalize, and format images to match input requirements for GAN-based models. 
**Text:** Clean, tokenize, and remove noise (e.g., emojis, excessive punctuation). 
**Prompt Conditioning Strategy:** We will preprocess captions by appending a prefix indicating mood or intent (e.g., “happy:”, “sarcastic:”) so that the model learns to align generated captions with the user’s input. 
**Dataset Splitting:** Split datasets into training (70%), validation (15%), and testing (15%) sets to support model training and evaluation. 

**3. Model Development** 

**Cartoonization Module:** We plan to implement a GAN-based image-to-cartoon transformation model. Pretrained models like CartoonGAN or CycleGAN may be fine-tuned for better alignment with meme aesthetics and user-uploaded selfies. 

**Caption Generation Module:** GPT-2 will be fine-tuned on mood-labeled captions to improve style, tone, and relevance. We will also evaluate the zero-shot baseline to compare prompt-only vs. fine-tuned performance. 

# Tools and Libraries 

- Python for full pipeline development 
- PyTorch or TensorFlow for model handling 
- Hugging Face Transformers for GPT-2 
- PIL or OpenCV for caption overlay 
- NumPy/Pandas for preprocessing and evaluation 

**4. System Integration**

Once both modules are independently validated, we will integrate them into a seamless, end-to-end system. The flow will include: 

- User uploads image and selects or types a mood/message 
- Image is cartoonized by the GAN model 
- Prompt is passed to the GPT-2 model to generate a caption 
- Caption is overlaid on the image using PIL, producing a stylized cartoon image with a personalized caption 
- Outputs will be saved as standard image formats for easy sharing or further customization. 

**5. Evaluation and Iteration**

We will assess performance through: 

**Caption-Prompt Alignment:** Using semantic similarity and internal accuracy checks to evaluate how well captions reflect the intended mood. 

**Model Benchmarking:** Comparing fine-tuned GPT-2 with zero-shot output in terms of fluency, relevance, and alignment. 

**Qualitative Review:** Manually inspecting generated images for humor, coherence, and visual-text harmony. 

**Error Logging:** Confusion matrix-style summaries and prompt-level performance tracking. 

 

**6. Expected Outcomes** 

We aim to develop a fully functional and user-friendly Python-based tool that enables users to generate personalized cartoon-style images with expressive captions. Users will upload an image and provide a short text prompt or mood. The system will produce a cartoonized version of the image and generate a corresponding caption using Generative AI. As a potential extension, we may explore integrating camera input from mobile or desktop devices for real-time image capture. 

Our expected outcomes include the following components: 

**Cartoonization Pipeline:**  
Converts user-uploaded images into high-quality cartoon-style visuals using a pretrained CartoonGAN model 
Preserves key facial features and maintains artistic consistency 

**Image Captioning:**  
Fine-tuned GPT-2 model generates expressive, humorous, and context-aware captions 
Captions are based on the user's provided mood or message 
Demonstrate NLP’s role in creative text generation task 

**Integrated Captioned Image Generator:**  
Seamless end-to-end pipeline combining cartoonization and caption generation 
Delivers coherent and visually aligned images that reflect user prompts through effective image-caption fusion 

**Performance Evaluation:**  
Measures caption quality and relevance using proper evaluation tools, such as semantic similarity in embedding space 
Uses confusion matrices and prompt-alignment accuracy metrics to assess alignment with user mood 
Includes qualitative analysis of visual-text coherence and stylistic consistency 

**Model Comparison and Evaluation Strategy:**  
As an important learning target, compares outputs from fine-tuned GPT-2 with baseline (zero-shot) GPT-2 
Helps benchmark the benefits of fine-tuning and informs future improvement 

**Further Potential Impact**

**Enhancing Creativity:** The tool enables non-technical users to express themselves creatively using Generative AI, helping overcome the technical and artistic barriers to meme creation and visual storytelling. 
**Showcasing Generative AI Capabilities:** This project serves as a practical example of combining computer vision and natural language processing to generate personalized content, demonstrating the real-world potential of generative AI in media, entertainment, and digital communication. 
**Foundation for Future Extensions:** The system will be designed with scalability in mind, providing a strong base for future development. It could be extended into a standalone mobile application or integrated as a feature within existing social media, messaging, or content creation platforms. 

 

**7. Team Roles**

To ensure the successful implementation of the project, each team member is assigned clear responsibilities across different phases of development. Roles are distributed based on individual strengths and interests, supporting effective collaboration, knowledge sharing, and smooth progress through research, model development, evaluation, and integration. 

**Shan** will conduct research on various model architectures and approaches relevant to both cartoonization and text generation. Also handles identifying and compiling suitable reference datasets required for fine-tuning the pretrained models, ensuring alignment with industry best practices. 

**Arish** will select the most appropriate models for cartoonization and caption generation. Responsible for conducting comprehensive testing on similar existing pretrained models and developing a functional Minimum Viable Product (MVP). This MVP will serve as the foundation for further assessment, refinement, and alignment with specific project needs. 

**Paolo** focuses on performance evaluation and metric design. Responsible for analyzing the quality and consistency of generated outputs using tools such as confusion matrices and accuracy metrics, particularly to assess how well the generated captions align with the intended mood or prompt input. 

**Patricia** (role yet to be finalized due to time zone difference): May assist in later stages with testing, documentation, or coordination tasks, depending on project needs. 

 

**8. Next Steps for Future Milestones**

**Weeks 3–4:** Feedback & Early Exploration 

Submit initial project proposal and revise based on instructor feedback. 
Perform a literature review on CartoonGAN, GPT-2, text generation techniques, and cartoonized image generation approaches. 
Explore the datasets and assess their suitability. 
Set up project repository and define collaboration workflow (e.g., GitHub). 
Explore pretrained models and test environment setup for image transformation. 

**Weeks 5–8:** Prototyping & Data Preparation 

Perform EDA on selected datasets and identify necessary preprocessing steps (e.g., formatting, cleaning, and normalization). 
Implement a user input system for uploading images and entering mood/prompt. 
Begin experimenting with GPT-2 for text generation using fixed prompts. 
Build early end-to-end prototype (image → cartoon + fixed caption). 
Plan for caption overlay using Python Imaging Library (PIL). 

**Week 9 Milestone 1:** Final Project Proposal Submission 

Submit refined project proposal based on practical insights. Finalize dataset selection, model choice, and system architecture decisions. 
Update revised methodology and system architecture diagram accordingly. 
Clearly define evaluation criteria (confusion metrics, caption relevance, etc.). 

**Weeks 10–12 Milestone 2, 3:** Model Pipeline Implementation, Integration and Testing 

Develop and integrate a robust data pipeline for handling image inputs, validation, and preprocessing. 
Link mood/prompt inputs with caption generation logic. 
Finalize model integration, including refining CartoonGAN outputs and improving GPT-2 responses based on user prompts. 
Overlay generated captions on cartoonized images using PIL. 
Test integration of all modules (image, text, overlay) and conduct preliminary evaluation of generated image quality and caption relevance. 

**Week 13–14 Milestone 4** – Finalization and Submission 

Analyze results and evaluate captions (qualitative and/or quantitative metrics). 
Finalize project documentation, visuals, and presentation material. 
Submit completed system, final report, and group presentation. 

 
