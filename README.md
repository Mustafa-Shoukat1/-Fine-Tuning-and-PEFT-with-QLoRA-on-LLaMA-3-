<div style="position: relative; text-align: center; background-image: url('https://media.licdn.com/dms/image/D4D35AQHj1_GYdHxDig/profile-framedphoto-shrink_400_400/0/1718974840314?e=1720130400&v=beta&t=87mIwnblVeilRNWe6W2wAKfCsKKq-LJADFMm5yarCUI'); background-size: 70%; background-position: center; border-radius: 20px; border: 2px solid #64B5F6; padding: 15px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.19); transform: perspective(1000px) rotateX(5deg) rotateY(-5deg); transition: transform 0.5s ease-in-out;">
    <div style="position: relative; z-index: 1; background-color: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px;">
        <h1 style="color: red; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px;">Welcome!</h1>
        <p style="color: #1976D2; font-size: 18px; margin: 10px 0;">
            I'm Mustafa Shoukat, a data scientist. I'm in the world of AI and exploring various concepts and techniques to enhance my skills. In this notebook, I'll unlock the potential of your models with precision and efficiency.
        </p>
        <p style="color: #37983B; font-size: 16px; font-style: italic; margin: 10px 0;">
            "Community empowers growth through shared knowledge and mutual support."
        </p>
        <p style="color: #2980B9; font-size: 16px; font-style: italic; margin: 10px 0;">
            <strong>About Notebook:</strong> 🧠 Fine-Tuning and PEFT with QLoRA on LLaMA 3
        </p>
        <p style="color: #27AE60; font-size: 16px; font-style: italic; margin: 10px 0;">
            This notebook delves into the advanced techniques of fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) using QLoRA on the LLaMA 3 model. Designed for data science enthusiasts and professionals, it offers hands-on experience and practical insights to enhance model performance with efficient use of computational resources. Join the journey to master these cutting-edge techniques and elevate your machine learning projects.
        </p>
        <h2 style="color: red; margin-top: 15px; font-size: 28px;">Contact Information</h2>
        <table style="width: 100%; margin-top: 15px; border-collapse: collapse;">
            <tr style="background-color: #64B5F6; color: #ffffff;">
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">Name</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">Email</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">LinkedIn</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">GitHub</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">Kaggle</th>
            </tr>
            <tr style="background-color: #FFFFFF; color: #000000;">
                <td style="padding: 8px;">Mustafa Shoukat</td>
                <td style="padding: 8px;">mustafashoukat.ai@gmail.com</td>
                <td style="padding: 8px;">
                    <a href="https://www.linkedin.com/in/mustafashoukat/" target="_blank">
                        <img src="https://img.shields.io/badge/LinkedIn-0e76a8.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
                <td style="padding: 8px;">
                    <a href="https://github.com/Mustafa-Shoukat1" target="_blank">
                        <img src="https://img.shields.io/badge/GitHub-171515.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
                <td style="padding: 8px;">
                    <a href="https://www.kaggle.com/mustafashoukat" target="_blank">
                        <img src="https://img.shields.io/badge/Kaggle-20beff.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
            </tr>
        </table>
    </div>
</div>


<div style="position: relative; z-index: 1; background-color: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px;">
    <h1 style="color: red; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px; text-align: center;"></h1>
    <p style="color: #1976D2; font-size: 18px; margin: 10px 0; text-align: center;">
        <img src="https://miro.medium.com/v2/resize:fit:1200/1*rOW5plKBuMlGgpD0SO8nZA.png" alt="ORPO Diagram" style="display: block; margin: 0 auto; max-width: 100%; height: auto;"/>
    </p>
</div>


<div style="position: relative; text-align: center; background-image: url('https://media.licdn.com/dms/image/D4D35AQHj1_GYdHxDig/profile-framedphoto-shrink_400_400/0/1718974840314?e=1720130400&v=beta&t=87mIwnblVeilRNWe6W2wAKfCsKKq-LJADFMm5yarCUI'); background-size: 70%; background-position: center; border-radius: 20px; border: 2px solid #64B5F6; padding: 15px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.19); transform: perspective(1000px) rotateX(5deg) rotateY(-5deg); transition: transform 0.5s ease-in-out;">
    <div style="position: relative; z-index: 1; background-color: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px;">
        <h1 style="color: red; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px;">Welcome!</h1>
        <p style="color: #1976D2; font-size: 18px; margin: 10px 0;">
            I'm Mustafa Shoukat, a data scientist. I'm in the world of AI and exploring various concepts and techniques to enhance my skills. In this notebook, I'll unlock the potential of your models with precision and efficiency.
        </p>
        <p style="color: #37983B; font-size: 16px; font-style: italic; margin: 10px 0;">
            "Community empowers growth through shared knowledge and mutual support."
        </p>
        <p style="color: #2980B9; font-size: 16px; font-style: italic; margin: 10px 0;">
            <strong>About Notebook:</strong> 🧠 Fine-Tuning and PEFT with QLoRA on LLaMA 3
        </p>
        <p style="color: #27AE60; font-size: 16px; font-style: italic; margin: 10px 0;">
            This notebook delves into the advanced techniques of fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) using QLoRA on the LLaMA 3 model. Designed for data science enthusiasts and professionals, it offers hands-on experience and practical insights to enhance model performance with efficient use of computational resources. Join the journey to master these cutting-edge techniques and elevate your machine learning projects.
        </p>
        <h2 style="color: red; margin-top: 15px; font-size: 28px;">Contact Information</h2>
        <table style="width: 100%; margin-top: 15px; border-collapse: collapse;">
            <tr style="background-color: #64B5F6; color: #ffffff;">
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">Name</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">Email</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">LinkedIn</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">GitHub</th>
                <th style="padding: 8px; border-bottom: 2px solid #1976D2;">Kaggle</th>
            </tr>
            <tr style="background-color: #FFFFFF; color: #000000;">
                <td style="padding: 8px;">Mustafa Shoukat</td>
                <td style="padding: 8px;">mustafashoukat.ai@gmail.com</td>
                <td style="padding: 8px;">
                    <a href="https://www.linkedin.com/in/mustafashoukat/" target="_blank">
                        <img src="https://img.shields.io/badge/LinkedIn-0e76a8.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
                <td style="padding: 8px;">
                    <a href="https://github.com/Mustafa-Shoukat1" target="_blank">
                        <img src="https://img.shields.io/badge/GitHub-171515.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
                <td style="padding: 8px;">
                    <a href="https://www.kaggle.com/mustafashoukat" target="_blank">
                        <img src="https://img.shields.io/badge/Kaggle-20beff.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge" style="border-radius: 5px; width: 100px;">
                    </a>
                </td>
            </tr>
        </table>
    </div>
</div>


<div style="position: relative; z-index: 1; background-color: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px;">
    <h1 style="color: red; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px; text-align: center;"></h1>
    <p style="color: #1976D2; font-size: 18px; margin: 10px 0; text-align: center;">
        <img src="https://miro.medium.com/v2/resize:fit:1200/1*rOW5plKBuMlGgpD0SO8nZA.png" alt="ORPO Diagram" style="display: block; margin: 0 auto; max-width: 100%; height: auto;"/>
    </p>
</div>

<div style="position: relative; text-align: center; background-image: url('https://th.bing.com/th/id/OIP.zzdnmTrMMKuSlTl7PPSZWwHaE8?rs=1&pid=ImgDetMain'); background-size: 70%; background-position: center; border-radius: 20px; border: 2px solid #64B5F6; padding: 15px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.19); transform: perspective(1000px) rotateX(5deg) rotateY(-5deg); transition: transform 0.5s ease-in-out;">
    <div style="position: relative; z-index: 1; background-color: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px;">
        <h1 style="color: red; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); font-weight: bold; margin-bottom: 10px; font-size: 32px;">Traditional vs New Transfer learning</h1>
        <p style="color: #1976D2; font-size: 18px; margin: 10px 0;">

### Traditional Transfer Learning
- **Uses:** A pre-trained model (weights and architecture) trained on a large dataset (source task) for a new task (target task).
- **Key Point:** Freezes most of the pre-trained model's parameters, only training a small subset (usually the final layers) on the new data.
- **Benefit:** Faster training, leverages learned features, good for tasks with limited data.

### Fine-Tuning
- **Also leverages:** A pre-trained model for a new task.
- **Key Point:** Trains all the parameters of the pre-trained model on the new data.
- **Benefit:** Allows for more significant adaptation to the new task, potentially higher performance compared to transfer learning.
- **Drawback:** Can be computationally expensive, potentially prone to overfitting with small datasets.

### Parameter-Efficient Fine-Tuning (PEFT)
- **A specific strategy:** Within fine-tuning.
- **Key Point:** Focuses on training only a subset of the pre-trained model's parameters, often identified through techniques like saliency maps or gradient analysis.
- **Benefit:** Reduces computational cost compared to full fine-tuning while achieving comparable performance in some cases.

### Adapter
Adapters are a special type of submodule that can be added to pre-trained language models to modify their hidden representation during fine-tuning. By inserting adapters after the multi-head attention and feed-forward layers in the transformer architecture, we can update only the parameters in the adapters during fine-tuning while keeping the rest of the model parameters frozen.

Adopting adapters can be a straightforward process. All that is required is to add adapters into each transformer layer and place a classifier layer on top of the pre-trained model. By updating the parameters of the adapters and the classifier head, we can improve the performance of the pre-trained model on a particular task without updating the entire model. This approach can save time and computational resources while still producing impressive results.

### LoRA
Low-rank adaptation (LoRA) of large language models is another approach in the area of fine-tuning models for specific tasks or domains. Similar to the adapters, LoRA is also a small trainable submodule that can be inserted into the transformer architecture. It involves freezing the pre-trained model weights and injecting trainable rank decomposition matrices into each layer of the transformer architecture, greatly diminishing the number of trainable parameters for downstream tasks. This method can minimize the number of trainable parameters by up to 10,000 times and the GPU memory necessity by 3 times while still performing on par or better than fine-tuning model quality on various tasks. LoRA also allows for more efficient task-switching, lowering the hardware barrier to entry, and has no additional inference latency compared to other methods.

### Fine-tune Llama 3 with ORPO
ORPO is a new exciting fine-tuning technique that combines the traditional supervised fine-tuning and preference alignment stages into a single process. This reduces the computational resources and time required for training. Moreover, empirical results demonstrate that ORPO outperforms other alignment methods on various model sizes and benchmarks.

There are now many methods to align large language models (LLMs) with human preferences. Reinforcement learning with human feedback (RLHF) was one of the first and brought us ChatGPT, but RLHF is very costly. DPO (Differentiable Preference Optimization), IPO (Interactive Preference Optimization), and KTO (Knowledge Transfer Optimization) are notably cheaper than RLHF as they don’t need a reward model.

While DPO and IPO are cheaper, they still require to train two different models. One model for the supervised fine-tuning (SFT) step, i.e., training the model to answer instructions, and then the model to align with human preferences using the SFT model for initialization and as a reference.

### ORPO
Instruction tuning and preference alignment are essential techniques for adapting Large Language Models (LLMs) to specific tasks. Traditionally, this involves a multi-stage process: 1/ Supervised Fine-Tuning (SFT) on instructions to adapt the model to the target domain, followed by 2/ preference alignment methods like Reinforcement Learning with Human Feedback (RLHF) or Direct Preference Optimization (DPO) to increase the likelihood of generating preferred responses over rejected ones.

ORPO is yet another new method for LLM alignment but this one doesn’t even need the SFT model. With ORPO, the LLM jointly learns to answer instructions and human preferences.
ORPO: Monolithic Preference Optimization without Reference Model
        </p>
    </div>
</div>

![Transfer Learning Image](https://assets.isu.pub/document-structure/230601111519-67cde4fe1c7eba3b19bcb148f484d14a/v1/5875b3596bd87a85183be5a114dd4fd0.jpeg)

