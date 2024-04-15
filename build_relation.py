from model import SCPNet, load_clip_to_cpu, TextEncoder, PromptLearner
from config import cfg
import torch.nn.functional as F
import numpy as np

def encode_by_clip_text_encoder(classnames): 
    clip_model = load_clip_to_cpu()
    clip_model.float()

    prompt_learner = PromptLearner(classnames, clip_model)
    tokenized_prompts = prompt_learner.tokenized_prompts
    text_encoder = TextEncoder(clip_model)
    logit_scale = clip_model.logit_scale
    logit_scale = logit_scale.exp()

    prompt_learner = PromptLearner(classnames, clip_model)
    prompts = prompt_learner()
    text_features = text_encoder(prompts, tokenized_prompts)
    
    return text_features

def compute_similarity_matrix(text_features):
    text_features = F.normalize(text_features, dim=1)
    labels_similarity = F.cosine_similarity(text_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
    return labels_similarity

if __name__ == '__main__':

    # Read labels from file
    try:
        with open('voc_labels.txt', 'r') as f:
            text = f.read()
            classnames = text.split('\n') 
        print("Number of classes: ", len(classnames))
    except:
        print()
        print("Cannot find the glacemood_labels.txt file. Please run format_glacemood.py first")
        exit()


    text_features = encode_by_clip_text_encoder(classnames)
    print("Text features shape: ", text_features.shape)

    # Compute Cosine Similarity between text features

    labels_similarity = compute_similarity_matrix(text_features)

    print("Similarity matrix shape: ", labels_similarity.shape)

    # print(labels_similarity)

    # save the matrix into npy form 
    np.save('relation+voc_vit_b32.npy', labels_similarity.detach().numpy())
    print("Saved the similarity matrix into npy file")