import os
import re

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from AdaDC_Dataset import TensorDatasetWithIndices, DatasetWithIndices


# class AdaDC():
#     def __init__(self, args, noisy_dataset, model):
#         self.args = args
#         self.model_name = args.model_name
#         self.seed = args.seed
        
#         self.dynamic_corrected_dataset = copy.deepcopy(noisy_dataset)
#         self.lastlayer = model.get_lastlayer()
        
#         # init data split
#         self.filter_clean_ids = []
#         self.hard_relabel_ids = []
#         self.corrected_labels = None
    
def compute_entropy(prob_matrix):
    # Add a small epsilon to avoid log(0) issues
    epsilon = 1e-10
    prob_matrix = np.clip(prob_matrix, epsilon, 1)
    # Compute entropy for each sample
    entropy = -np.sum(prob_matrix * np.log(prob_matrix), axis=1)
    return entropy   

def cross_entropy(p, q):
    return -np.sum(p * np.log(q + 1e-9), axis=1)  # Adding epsilon to avoid log(0)

def get_emb_dataloader(model, loader, batch_size=256, f_dim=512, device='cuda'):
    # model.load_state_dict(torch.load(ckpt_path)['net'])
    model = model.to(device)
    model.train()
    emb_tensors = torch.zeros(len(loader.dataset), f_dim)
    target_tensors = torch.zeros(len(loader.dataset), dtype=torch.long)
    with torch.no_grad():
        for batch, indices in tqdm(loader, total=len(loader)):
            data, target = batch
            # if data is a list
            if isinstance(data, list):
                data, target = data[0].to(device), target.to(device)
            else:
                data, target = data.to(device), target.to(device)
            # print(data.shape)
            emb = model.extract_embedding(data)
            emb_tensors[indices] = emb.detach().cpu()
            target_tensors[indices] = target.detach().cpu()
    emb_tensor_dataset = torch.utils.data.TensorDataset(emb_tensors, target_tensors)
    emb_loader = DataLoader(TensorDatasetWithIndices(emb_tensor_dataset), batch_size=batch_size, shuffle=False)
    return emb_loader


def get_influence_scores_batch(model, train_emb_batch, val_emb_batch, criterion, device='cuda', num_classes=10, lr=None):
    
    lr = lr if lr is not None else 0.01
    train_emb, train_target, allone_train_target = train_emb_batch[0].to(device), train_emb_batch[1].to(device), torch.ones((train_emb_batch[1].shape[0], num_classes)).to(device)
    val_emb, val_target = val_emb_batch[0].to(device), val_emb_batch[1].to(device)
    
    # get last layer
    lastlayer = model.get_last_layer()
    lastlayer = lastlayer.to(device)
    optimizer = torch.optim.SGD(lastlayer.parameters(), lr=lr)
    optimizer.zero_grad()
    train_output = lastlayer(train_emb.detach())
    train_loss = criterion(train_output, train_target)
    train_loss = train_loss.mean()
    train_loss.backward()
    optimizer.step()
    # b_s = train_emb.size(0)

    
    # To handle multiple layers and blocks
    # grads = []
    # for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
    #     for block in layer:
    #         if hasattr(block.conv1, 'grad_sample'):
    #             grad = block.conv1.grad_sample.reshape(b_s, -1).detach()  # Detach each gradient
    #             # print("conv grad:##########")
    #             # print(grad.shape)
    #             grads.append(grad)
    # fc_grad = lastlayer.fc.grad_sample.reshape(b_s, -1).detach()
    # grads.append(fc_grad)
    # train_grad = torch.cat(grads, dim=1)
    train_grad = lastlayer.get_persample_grad()
    
    
    # get allone train grads
    lastlayer = model.get_last_layer()
    lastlayer = lastlayer.to(device)
    optimizer.zero_grad()
    train_output = lastlayer(train_emb.detach())
    train_loss = criterion(train_output, allone_train_target)
    train_loss = train_loss.mean()
    train_loss.backward()
    optimizer.step()
    # b_s = train_emb.size(0)
    
    # To handle multiple layers and blocks
    # grads = []
    # for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
    #     for block in layer:
    #         if hasattr(block.conv1, 'grad_sample'):
    #             grad = block.conv1.grad_sample.reshape(b_s, -1).detach()  # Detach each gradient
    #             # print("conv grad:##########")
    #             # print(grad.shape)
    #             grads.append(grad)
    # fc_grad = lastlayer.fc.grad_sample.reshape(b_s, -1).detach()
    # grads.append(fc_grad)
    # allone_train_grad = torch.cat(grads, dim=1)
    allone_train_grad = lastlayer.get_persample_grad()
    
    
    # get val grads
    lastlayer = model.get_last_layer()
    optimizer.zero_grad()
    val_output = lastlayer(val_emb.detach())
    val_loss = criterion(val_output, val_target)
    val_loss = val_loss.mean()
    val_loss.backward()
    optimizer.step()
    # b_s = val_emb.size(0)
    
    # grads = []
    # for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
    #     for block in layer:
    #         if hasattr(block.conv1, 'grad_sample'):
    #             grad = block.conv1.grad_sample.reshape(b_s, -1).detach()  # Detach each gradient
    #             # print("conv grad:##########")
    #             # print(grad.shape)
    #             grads.append(grad)
    # fc_grad = lastlayer.fc.grad_sample.reshape(b_s, -1).detach()
    # grads.append(fc_grad)
    # val_grad = torch.cat(grads, dim=1)
    val_grad = lastlayer.get_persample_grad()
    
    # graddot = (train_grad @ val_grad.T).detach().cpu().numpy()
    # embdot = (train_emb @ val_emb.T).detach().cpu().numpy()
    # allone_graddot = (allone_train_grad @ val_grad.T).detach().cpu().numpy()
    
    # influence_scores = embdot * graddot
    # allone_influence_scores = embdot * allone_graddot
    
    
    # compute cosine similarity
    def compute_cosine_similarity(train_vectors, val_vectors, epsilon=1e-8):
        """
        Computes the cosine similarity matrix between two sets of vectors.

        Args:
        - train_vectors (torch.Tensor): A tensor of shape [train num, dim] representing training vectors.
        - val_vectors (torch.Tensor): A tensor of shape [val num, dim] representing validation vectors.
        - epsilon (float): A small value to replace zero norms and avoid division by zero.

        Returns:
        - torch.Tensor: A cosine similarity matrix of shape [train num, val num].
        """

        # Compute norms
        train_norms = train_vectors.norm(dim=1, keepdim=True)
        val_norms = val_vectors.norm(dim=1, keepdim=True)

        # Replace zero norms with a small value (epsilon) to avoid division by zero
        train_norms = torch.where(train_norms == 0, torch.tensor(epsilon, device=train_vectors.device), train_norms)
        val_norms = torch.where(val_norms == 0, torch.tensor(epsilon, device=val_vectors.device), val_norms)

        # Normalize each vector in the training and validation sets to have unit norm
        train_norm = train_vectors / train_norms
        val_norm = val_vectors / val_norms

        # Compute the dot product between all pairs of normalized vectors
        cosine_similarity_matrix = torch.mm(train_norm, val_norm.T)

        return cosine_similarity_matrix
    
    
    grad_cossim = compute_cosine_similarity(train_grad, val_grad).detach().cpu().numpy()
    allone_grad_cossim = compute_cosine_similarity(allone_train_grad, val_grad).detach().cpu().numpy()
    emb_cossim = compute_cosine_similarity(train_emb, val_emb).detach().cpu().numpy()
    # print(grad_cossim.shape)
    # print(emb_cossim.shape)
    
    influence_scores = emb_cossim * grad_cossim
    allone_influence_scores = emb_cossim * allone_grad_cossim
    
    # free memory
    del train_output, train_loss, train_grad, val_output, val_loss, val_grad, allone_train_grad
    torch.cuda.empty_cache()
    
    return influence_scores, allone_influence_scores
    



def get_influence_scores(model, target_train_loader, val_loader, criterion, ckpt_list, num_classes=10, f_dim=512):
    inf_list = []
    allone_inf_list = []
    lr_list = []
    emb_batch_size = 1000
    for ckpt in ckpt_list:
        print("Computing influence scores for ckpt: ", ckpt)
        
        #load model
        # model.load_state_dict(torch.load(ckpt)['teacher_state_dict'])
        # model.load_state_dict(torch.load(ckpt, weights_only=True)['state_dict'])
        # model.load_state_dict(torch.load(ckpt))
        # match checkpoint key name
        checkpoint = torch.load(ckpt)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'teacher_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['teacher_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        if 'optimizer' in checkpoint:
            # get lr
            # Load the optimizer state dict
            optimizer_state_dict = checkpoint['optimizer']
            # Extract the learning rate from the optimizer's param_groups
            learning_rates = [param_group['lr'] for param_group in optimizer_state_dict['param_groups']]
            print(learning_rates[-1])
            # lr_list.append(learning_rates[-1])
            lr_list.append(1)
        else:
            lr_list.append(1)

            
            
        
        # get emb tensor dataloader for hard train and easy test
        print("Extracting embeddings...")
        hard_train_emb_loader = get_emb_dataloader(model, target_train_loader, batch_size=emb_batch_size, f_dim=f_dim)
        hard_test_emb_loader = get_emb_dataloader(model, val_loader, batch_size=emb_batch_size, f_dim=f_dim)
        inf_scores = torch.zeros(len(target_train_loader.dataset), len(val_loader.dataset))
        allone_inf_scores = torch.zeros(len(target_train_loader.dataset), len(val_loader.dataset))
        
        print("Computing influence scores...")
        for train_emb_batch, train_indices in tqdm(hard_train_emb_loader):
            for val_emb_batch, val_indices in hard_test_emb_loader:
                influence_scores_batch, allone_influence_scores_batch = get_influence_scores_batch(model, train_emb_batch, val_emb_batch, criterion, num_classes=num_classes)
                inf_scores[train_indices[:, None], val_indices[None, :]] = torch.tensor(influence_scores_batch)
                allone_inf_scores[train_indices[:, None], val_indices[None, :]] = torch.tensor(allone_influence_scores_batch)
                # print("inf_scores:", inf_scores)
                # print("allone inf scores:", allone_inf_scores)
        inf_list.append(inf_scores.numpy())
        allone_inf_list.append(allone_inf_scores.numpy())
    
    # weighted sum by lr_list 
    inf_scores_weighted_sum = np.zeros((len(target_train_loader.dataset), len(val_loader.dataset)))
    allone_inf_scores_weighted_sum = np.zeros((len(target_train_loader.dataset), len(val_loader.dataset)))
    for i in range(len(ckpt_list)):
        inf_scores_weighted_sum += inf_list[i] * lr_list[i]
        allone_inf_scores_weighted_sum += allone_inf_list[i] * lr_list[i]
    
    inf_scores_sum = inf_scores_weighted_sum
    allone_inf_scores_sum = allone_inf_scores_weighted_sum
    
    # inf_scores_sum = np.sum(inf_list, axis=0)
    # print(inf_scores_sum.sum(axis=1))
    # print("inf_scores_sum", inf_scores_sum)

    val_targets = torch.zeros(len(val_loader.dataset), dtype=torch.long)
    for batch, val_indices in val_loader:
        val_targets[val_indices] = batch[1]
    val_targets = val_targets.numpy()

    inf_cls = np.zeros((inf_scores_sum.shape[0], num_classes))
    for i in range(num_classes):
        cls_ids = np.where(val_targets == i)[0]
        if len(cls_ids) == 0:
            print("No samples for class:", i)
        inf_cls[:, i] = np.mean(inf_scores_sum[:, cls_ids], axis=1)

    # find all positions where inf_cls is nan
    # nan_ids = np.where(np.isnan(inf_scores_sum))
    # print("inf_scores_sum nan_ids:", nan_ids)
    # print("labels for nan_ids:", val_targets[nan_ids[1]])
    # nan_ids = np.where(np.isnan(inf_cls))
    # print("inf_cls nan_ids:", nan_ids)

    # normalize
    # print("inf_cls before normalize:", inf_cls)
    
    
    
    # inf_cls = inf_cls / np.linalg.norm(inf_cls, axis=1, keepdims=True)

    # print("inf_cls after normalize:", inf_cls)

    # allone_inf_scores_sum = np.sum(allone_inf_list, axis=0)
    # print(inf_scores_sum.sum(axis=1))
    # print(allone_inf_scores_sum.shape)


    allone_inf_cls = np.zeros((allone_inf_scores_sum.shape[0], num_classes))
    for i in range(num_classes):
        cls_ids = np.where(val_targets == i)[0]
        if len(cls_ids) == 0:
            print("No samples for class:", i)
        allone_inf_cls[:, i] = np.mean(allone_inf_scores_sum[:, cls_ids], axis=1)

    # normalize
    # allone_inf_cls = allone_inf_cls / np.linalg.norm(allone_inf_cls, axis=1, keepdims=True)
    
    # Assuming inf_cls and allone_inf_cls are NumPy arrays
    epsilon = 1e-8  # Small value to prevent division by zero

    # Compute the norms
    inf_cls_norms = np.linalg.norm(inf_cls, axis=1, keepdims=True)
    allone_inf_cls_norms = np.linalg.norm(allone_inf_cls, axis=1, keepdims=True)

    # Replace zero norms with epsilon to avoid division by zero
    inf_cls_norms[inf_cls_norms == 0] = epsilon
    allone_inf_cls_norms[allone_inf_cls_norms == 0] = epsilon

    # Normalize the vectors
    inf_cls = inf_cls / inf_cls_norms
    allone_inf_cls = allone_inf_cls / allone_inf_cls_norms

    # print(allone_inf_cls.shape)
        
    return inf_cls, allone_inf_cls


        
def compute_noise_signal(inf_cls, allone_inf_cls, noisy_train_loader, train_clean_labels=None, train_hard_ids=None, num_classes=10, sig_type='sig12'):
    dataset_size = len(noisy_train_loader.dataset)
    train_noise_labels = torch.zeros(dataset_size, dtype=torch.long)
    for batch, indices in noisy_train_loader:
        train_noise_labels[indices] = batch[1]
    train_noise_labels = train_noise_labels.numpy()
    
    if train_clean_labels is not None:
        noise_ids = np.where(train_noise_labels != train_clean_labels)[0]
    else:
        noise_ids = None


    assigned_label = train_noise_labels
    assigned_inf = inf_cls[range(len(assigned_label)), assigned_label]
    # print(assigned_inf.shape)
    allone_assigned_inf = allone_inf_cls[range(len(assigned_label)), assigned_label]
    sig1 = abs(allone_assigned_inf)
    
    # print("inf_cls:", inf_cls)
    # print("allone_inf_cls:", allone_inf_cls)
    
    negative_inf_cls = -inf_cls
    negative_allone_inf_cls = -allone_inf_cls
    negative_inf_cls[negative_inf_cls < 0] = 0
    negative_allone_inf_cls[negative_allone_inf_cls < 0] = 0
    
    # print("negative_inf_cls:", negative_inf_cls)
    # print("negative_allone_inf_cls:", negative_allone_inf_cls)

    # negative_inf_cls = (negative_inf_cls - negative_inf_cls.min(axis=1, keepdims=True)) / (negative_inf_cls.max(axis=1, keepdims=True) - negative_inf_cls.min(axis=1, keepdims=True))
    # negative_allone_inf_cls = (negative_allone_inf_cls - negative_allone_inf_cls.min(axis=1, keepdims=True)) / (negative_allone_inf_cls.max(axis=1, keepdims=True) - negative_allone_inf_cls.min(axis=1, keepdims=True))

    # check if all zeros
    zero_ids = np.where(np.sum(negative_inf_cls, axis=1) == 0)[0]
    if len(zero_ids) > 0:
        print("inf Zero ids:", len(zero_ids))
        if noise_ids is not None:
            print("noise num in zero ids:", np.sum(np.isin(zero_ids, noise_ids)))
        if train_hard_ids is not None:
            print("hard num in zero ids:", np.sum(np.isin(zero_ids, train_hard_ids)))
        negative_inf_cls[zero_ids] = np.ones((len(zero_ids), num_classes))*1e-7 / num_classes
        
    zero_ids = np.where(np.sum(negative_allone_inf_cls, axis=1) == 0)[0]
    if len(zero_ids) > 0:
        print("allone inf Zero ids:", len(zero_ids))
        if noise_ids is not None:
            print("noise num in zero ids:", np.sum(np.isin(zero_ids, noise_ids)))
        if train_hard_ids is not None:
            print("hard num in zero ids:", np.sum(np.isin(zero_ids, train_hard_ids)))
        negative_allone_inf_cls[zero_ids] = np.ones((len(zero_ids), num_classes))*1e-7 / num_classes
        
    
    negative_inf_cls = negative_inf_cls / negative_inf_cls.sum(axis=1, keepdims=True)
    negative_allone_inf_cls = negative_allone_inf_cls / negative_allone_inf_cls.sum(axis=1, keepdims=True)


    nan_ids = np.where(np.isnan(negative_inf_cls))[0]
    print("negative_inf_cls nan_ids:", nan_ids)
    nan_ids = np.where(np.isnan(negative_allone_inf_cls))[0]
    print("negative_allone_inf_cls nan_ids:", nan_ids)
    
    def cross_entropy(p, q):
        return -np.sum(p * np.log(q + 1e-9), axis=1)  # Adding epsilon to avoid log(0)

    def kl_divergence(p, q):
        return np.sum(p * np.log((p + 1e-9) / (q + 1e-9)), axis=1)  # Adding epsilon to avoid division by zero and log(0)

    # compute the loss between negative_inf_cls and negative_allone_inf_cls
    sig2 = cross_entropy(negative_inf_cls, negative_allone_inf_cls)
    # print("sig2:", sig2)
    
    # check where sig2 is nan
    nan_ids = np.where(np.isnan(sig2))[0]
    print("sig2 nan_ids:", nan_ids)
    # set nan to 0.5
    sig2[nan_ids] = 0.5
    
    # print(sig2.min(), sig2.max())
    # normalize the sig to 0-1
    sig2 = (sig2 - sig2.min()) / (sig2.max() - sig2.min())
    
    
    
    def compute_entropy(prob_matrix):
        # Add a small epsilon to avoid log(0) issues
        epsilon = 1e-10
        prob_matrix = np.clip(prob_matrix, epsilon, 1)
        # Compute entropy for each sample
        entropy = -np.sum(prob_matrix * np.log(prob_matrix), axis=1)
        return entropy

    sig3 = compute_entropy(negative_allone_inf_cls)
    print("sig3:", sig3.shape)
    
    # print("inf_cls:",inf_cls)
    # print("allone_inf_cls:", allone_inf_cls)
    
    
    # print("sig1:", sig1)
    # print("sig2:", sig2)

    # sig = sig1 * sig2
    lambda1 = 0.8
    if sig_type == 'sig12':
        sig = lambda1 * sig1 + (1 - lambda1) * sig2
    elif sig_type == 'sig1':
        sig = sig1
    elif sig_type == 'sig2':
        sig = sig2
    elif sig_type == 'sig3':
        sig = sig3
    
    print("Noise sig shape:", sig.shape)
    
    labels_after_softLabel = negative_allone_inf_cls
    
    return sig, labels_after_softLabel


     
def compute_noise_signal_new(inf_cls, allone_inf_cls, noisy_train_loader, train_clean_labels=None, train_hard_ids=None, num_classes=10, sig_type='sig12'):
    dataset_size = len(noisy_train_loader.dataset)
    train_noise_labels = torch.zeros(dataset_size, dtype=torch.long)
    for batch, indices in noisy_train_loader:
        train_noise_labels[indices] = batch[1]
    train_noise_labels = train_noise_labels.numpy()
    
    if train_clean_labels is not None:
        noise_ids = np.where(train_noise_labels != train_clean_labels)[0]
    else:
        noise_ids = None


    assigned_label = train_noise_labels
    assigned_inf = inf_cls[range(len(assigned_label)), assigned_label]
    # print(assigned_inf.shape)
    allone_assigned_inf = allone_inf_cls[range(len(assigned_label)), assigned_label]
    sig1 = abs(allone_assigned_inf)
    
    # print("inf_cls:", inf_cls)
    # print("allone_inf_cls:", allone_inf_cls)
    
    negative_inf_cls = -inf_cls
    negative_allone_inf_cls = -allone_inf_cls
    negative_inf_cls[negative_inf_cls < 0] = 0
    negative_allone_inf_cls[negative_allone_inf_cls < 0] = 0
    
    # print("negative_inf_cls:", negative_inf_cls)
    # print("negative_allone_inf_cls:", negative_allone_inf_cls)

    # negative_inf_cls = (negative_inf_cls - negative_inf_cls.min(axis=1, keepdims=True)) / (negative_inf_cls.max(axis=1, keepdims=True) - negative_inf_cls.min(axis=1, keepdims=True))
    # negative_allone_inf_cls = (negative_allone_inf_cls - negative_allone_inf_cls.min(axis=1, keepdims=True)) / (negative_allone_inf_cls.max(axis=1, keepdims=True) - negative_allone_inf_cls.min(axis=1, keepdims=True))

    # check if all zeros
    zero_ids = np.where(np.sum(negative_inf_cls, axis=1) == 0)[0]
    if len(zero_ids) > 0:
        print("inf Zero ids:", len(zero_ids))
        if noise_ids is not None:
            print("noise num in zero ids:", np.sum(np.isin(zero_ids, noise_ids)))
        if train_hard_ids is not None:
            print("hard num in zero ids:", np.sum(np.isin(zero_ids, train_hard_ids)))
        negative_inf_cls[zero_ids] = np.ones((len(zero_ids), num_classes))*1e-7 / num_classes
        
    zero_ids = np.where(np.sum(negative_allone_inf_cls, axis=1) == 0)[0]
    if len(zero_ids) > 0:
        print("allone inf Zero ids:", len(zero_ids))
        if noise_ids is not None:
            print("noise num in zero ids:", np.sum(np.isin(zero_ids, noise_ids)))
        if train_hard_ids is not None:
            print("hard num in zero ids:", np.sum(np.isin(zero_ids, train_hard_ids)))
        negative_allone_inf_cls[zero_ids] = np.ones((len(zero_ids), num_classes))*1e-7 / num_classes
        
    
    negative_inf_cls = negative_inf_cls / negative_inf_cls.sum(axis=1, keepdims=True)
    negative_allone_inf_cls = negative_allone_inf_cls / negative_allone_inf_cls.sum(axis=1, keepdims=True)


    nan_ids = np.where(np.isnan(negative_inf_cls))[0]
    print("negative_inf_cls nan_ids:", nan_ids)
    nan_ids = np.where(np.isnan(negative_allone_inf_cls))[0]
    print("negative_allone_inf_cls nan_ids:", nan_ids)
    
    def cross_entropy(p, q):
        return -np.sum(p * np.log(q + 1e-9), axis=1)  # Adding epsilon to avoid log(0)

    def kl_divergence(p, q):
        return np.sum(p * np.log((p + 1e-9) / (q + 1e-9)), axis=1)  # Adding epsilon to avoid division by zero and log(0)

    # compute the loss between negative_inf_cls and negative_allone_inf_cls
    sig2 = cross_entropy(negative_inf_cls, negative_allone_inf_cls)
    # print("sig2:", sig2)
    
    # check where sig2 is nan
    nan_ids = np.where(np.isnan(sig2))[0]
    print("sig2 nan_ids:", nan_ids)
    # set nan to 0.5
    sig2[nan_ids] = 0.5
    
    # print(sig2.min(), sig2.max())
    # normalize the sig to 0-1
    sig2 = (sig2 - sig2.min()) / (sig2.max() - sig2.min())
    
    
    
    def compute_entropy(prob_matrix):
        # Add a small epsilon to avoid log(0) issues
        epsilon = 1e-10
        prob_matrix = np.clip(prob_matrix, epsilon, 1)
        # Compute entropy for each sample
        entropy = -np.sum(prob_matrix * np.log(prob_matrix), axis=1)
        return entropy

    sig3 = compute_entropy(negative_allone_inf_cls)
    print("sig3:", sig3.shape)
    
    # print("inf_cls:",inf_cls)
    # print("allone_inf_cls:", allone_inf_cls)
    
    
    # print("sig1:", sig1)
    # print("sig2:", sig2)

    # sig = sig1 * sig2
    lambda1 = 0.8
    if sig_type == 'sig12':
        sig = lambda1 * sig1 + (1 - lambda1) * sig2
    elif sig_type == 'sig1':
        sig = sig1
    elif sig_type == 'sig2':
        sig = sig2
    elif sig_type == 'sig3':
        sig = sig3
    
    print("Noise sig shape:", sig.shape)
    
    labels_after_softLabel = negative_allone_inf_cls
    
    return sig, labels_after_softLabel


def plot_sig(sig):
    import matplotlib.pyplot as plt
    # clean_hard_ids = np.array(list(set(train_hard_ids) - set(noise_ids)))
    # clean_ids = np.array(list(set(range(len(train_dataset))) - set(noise_ids) - set(train_hard_ids)))
    # print(clean_hard_ids.shape, clean_ids.shape, sig.shape)
    # print(sig)
    plt.figure(figsize=(5, 3))
    # plt.hist(sig[clean_ids], bins=100, alpha=0.5, label='Clean Easy')
    # plt.hist(sig[noise_ids], bins=100, alpha=0.5, label='Noise')
    # plt.hist(sig[clean_hard_ids], bins=100, alpha=0.5, label='Clean Hard')
    plt.hist(sig, bins=100, alpha=0.5)
    plt.legend()
    plt.show()
        
        
def noise_detection(sig, labels_after_softLabel, noisy_train_loader, train_clean_labels=None, train_hard_ids=None, full_probs=None, detection_threshold=0.8, hard_relabel_threshold=0.8, num_classes=10, batch_size=256):
    dataset_size = len(noisy_train_loader.dataset)
    noisy_train_dataset = noisy_train_loader.dataset
    train_noise_labels = torch.zeros(dataset_size, dtype=torch.long)
    for batch, indices in noisy_train_loader:
        train_noise_labels[indices] = batch[1]
    train_noise_labels = train_noise_labels.numpy()
    
    if train_clean_labels is not None:
        noise_ids = np.where(train_noise_labels != train_clean_labels)[0]
    else:
        noise_ids = None
    # noise detection
    filter_clean_ids = np.where(sig > detection_threshold)[0]
    filter_noise_ids = np.array(list(set(range(dataset_size)) - set(filter_clean_ids)))
    if noise_ids is not None:
        print("filtered data:", "noise_num:", np.sum(np.isin(filter_clean_ids, noise_ids)), "all:", len(filter_clean_ids), "noise rate:", np.sum(np.isin(filter_clean_ids, noise_ids)) / len(filter_clean_ids))
        print("detected noise:", "noise_num:", np.sum(np.isin(filter_noise_ids, noise_ids)), "all:", len(filter_noise_ids), "noise rate:", np.sum(np.isin(filter_noise_ids, noise_ids)) / len(filter_noise_ids))
    else:
        print("filtered data:", len(filter_clean_ids), "detected noise:", len(filter_noise_ids))
    

    
    if full_probs is not None:
        avg_probs = np.mean(full_probs, axis=0)

    # relabel noise
    correction_labels = np.zeros((dataset_size, num_classes))
    hard_relabel_ids = []
    soft_relabel_ids = []
    for idx in filter_noise_ids:

        soft_label = labels_after_softLabel[idx]
        entropy = -np.sum(soft_label * np.log2(soft_label, where=(soft_label > 0)))
        # print(entropy)
        # print(soft_label)
        if entropy < hard_relabel_threshold:
            if full_probs is not None:
                probs = avg_probs[idx]
                # if max(probs) > 0.98 and np.argmax(probs) == np.argmax(soft_label):
                if max(probs) > 0.98:
                    hard_relabel_ids.append(idx)
                    onehot_code = np.eye(num_classes)
                    correction_labels[idx] = onehot_code[np.argmax(probs)]
            else:
                hard_relabel_ids.append(idx)
                onehot_code = np.eye(num_classes)
                correction_labels[idx] = onehot_code[np.argmax(soft_label)]
        else:
            soft_relabel_ids.append(idx)
            correction_labels[idx] = soft_label
        # hard_relabel_ids.append(idx)
        # correction_labels[idx] = soft_label
        
    hard_relabel_ids = np.array(hard_relabel_ids)
    

    # check hard relabel acc
    if len(hard_relabel_ids) > 0:
        labels_after_hardRelabel = np.argmax(correction_labels[hard_relabel_ids], axis=1)
        print("hard relabel num:", labels_after_hardRelabel.shape)
        if train_clean_labels is not None:
            print("Hard relabel acc", np.sum(labels_after_hardRelabel == train_clean_labels[hard_relabel_ids]) / len(hard_relabel_ids))
            noise_ids_after_relabel = np.where(np.argmax(correction_labels, axis=1) != train_clean_labels)[0]
            # compute noise rate after relabel
            print("Noise rate before hard relabel:", len(noise_ids) / dataset_size, "Noise num:", len(noise_ids))
            print("Noise rate after hard relabel:", (len(noise_ids_after_relabel)) / dataset_size, "Noise num:", len(noise_ids_after_relabel))
            print("Hard relabel num:", len(hard_relabel_ids), "Soft relabel num:", len(soft_relabel_ids))
   
    if train_clean_labels is not None and len(soft_relabel_ids) > 0 and len(hard_relabel_ids) > 0:
        # top 2 noise rate
        labels_after_relabel = train_noise_labels.copy()
        onehot_code = np.eye(num_classes)
        labels_after_relabel = onehot_code[labels_after_relabel]
        labels_after_relabel[np.concatenate([hard_relabel_ids, soft_relabel_ids])] = correction_labels[np.concatenate([hard_relabel_ids, soft_relabel_ids])]
        top2_ids = np.argsort(labels_after_relabel, axis=1)[:, -2:]
        true_labels = train_clean_labels.reshape(-1)
        is_in_top2 = np.any(top2_ids == true_labels[:, None], axis=1)
        print("Top2 noise rate:", np.sum(is_in_top2) / len(top2_ids))

        top1_ids = np.argsort(labels_after_relabel, axis=1)[:, -1:]
        true_labels = train_clean_labels.reshape(-1)
        is_in_top1 = np.any(top1_ids == true_labels[:, None], axis=1)
        print("Top1 noise rate:", np.sum(is_in_top1) / len(top1_ids))
        
    import copy
    
    new_train_labels = train_noise_labels.copy()
    softnhard_relabel_ids = np.concatenate([hard_relabel_ids, soft_relabel_ids])
    onehot_code = np.eye(num_classes)
    new_train_labels = onehot_code[new_train_labels]
    new_train_labels[softnhard_relabel_ids] = correction_labels[softnhard_relabel_ids]
    # print(new_train_labels.shape)
    # np.concatenate([hard_relabel_ids, soft_relabel_ids])

    new_train_dataset = copy.deepcopy(noisy_train_dataset)
    new_train_dataset.targets = new_train_labels
    new_train_dataset.transform = noisy_train_dataset.dataset.transform

    # check number of each classes in hard_trainable_ids
    check_labels = np.argmax(new_train_labels, axis=1)
    
    # if train_dataset has classes attribute
    if hasattr(noisy_train_dataset, 'classes'):
        classes = noisy_train_dataset.classes
    else:
        classes = np.arange(num_classes)
        
    if train_clean_labels is not None:
        noise_ids_after_relabel = np.where(np.argmax(new_train_labels, axis=1) != train_clean_labels)[0]

    trainable_ids = np.array(list(set(filter_clean_ids) | set(softnhard_relabel_ids)))
    
    min_num = 1e9
    max_load_rate = 1
    for i in range(num_classes):
        ids = np.where(check_labels == i)[0]
        ids = ids[np.isin(ids, trainable_ids)]
        if len(ids) < min_num:
            min_num = len(ids)
        if train_clean_labels is not None:
            print("Class:", classes[i], "Num:", len(ids), "Noise rate", np.sum(np.isin(ids, noise_ids_after_relabel)) / len(ids), "Noise num:", np.sum(np.isin(ids, noise_ids_after_relabel)))
    print("Min num for softnhard relabel:", min_num)
    
    # sample balanced ids
    max_load_num = min_num * max_load_rate
    balance_softnhard_relabel_ids = []
    for i in range(num_classes):
        cls_ids = np.where(check_labels == i)[0]
        cls_ids = cls_ids[np.isin(cls_ids, trainable_ids)]
        if len(cls_ids) > max_load_num:
            balance_softnhard_relabel_ids.extend(np.random.choice(cls_ids, int(max_load_num), replace=False))
        else:
            balance_softnhard_relabel_ids.extend(cls_ids)
    
    softnhard_relabel_train_dataset = torch.utils.data.Subset(new_train_dataset, balance_softnhard_relabel_ids)
    softnhard_relabel_train_loader = torch.utils.data.DataLoader(DatasetWithIndices(softnhard_relabel_train_dataset), batch_size=batch_size, shuffle=True)

    hard_trainable_ids = np.array(list(set(filter_clean_ids) | set(hard_relabel_ids)))
    min_num = 1e9
    for i in range(num_classes):
        ids = np.where(check_labels == i)[0]
        ids = ids[np.isin(ids, hard_trainable_ids)]
        if len(ids) < min_num:
            min_num = len(ids)
        if train_clean_labels is not None:
            print("Class:", classes[i], "Num:", len(ids), "Noise rate", np.sum(np.isin(ids, noise_ids_after_relabel)) / len(ids), "Noise num:", np.sum(np.isin(ids, noise_ids_after_relabel)))
    print("Min num for hard relabel:", min_num)
    # random sample min_num samples from each class
    new_hard_trainable_ids = []
    # check_labels = np.argmax(new_train_labels, axis=1)
    max_load_num = min_num * 1.2
    for i in range(num_classes):
        cls_ids = np.where(check_labels == i)[0]
        cls_ids = cls_ids[np.isin(cls_ids, hard_trainable_ids)]
        if len(cls_ids) > max_load_num:
            new_hard_trainable_ids.extend(np.random.choice(cls_ids, int(max_load_num), replace=False))
        else:
            new_hard_trainable_ids.extend(cls_ids)
        
    # new_hard_trainable_ids = hard_trainable_ids
    
    hard_relabel_train_dataset = torch.utils.data.Subset(new_train_dataset, new_hard_trainable_ids)
    hard_relabel_train_loader = torch.utils.data.DataLoader(DatasetWithIndices(hard_relabel_train_dataset), batch_size=batch_size, shuffle=True)
    
    min_num = 1e9
    for i in range(num_classes):
        ids = np.where(check_labels == i)[0]
        ids = ids[np.isin(ids, filter_clean_ids)]
        if len(ids) < min_num:
            min_num = len(ids)
        if train_clean_labels is not None:
            print("Class:", classes[i], "Num:", len(ids), "Noise rate", np.sum(np.isin(ids, noise_ids_after_relabel)) / len(ids), "Noise num:", np.sum(np.isin(ids, noise_ids_after_relabel)))
    print("Min num for clean filter:", min_num)
    balance_filter_clean_ids = []
    max_load_num = min_num * 1.2
    for i in range(num_classes):
        cls_ids = np.where(check_labels == i)[0]
        cls_ids = cls_ids[np.isin(cls_ids, filter_clean_ids)]
        if len(cls_ids) > max_load_num:
            balance_filter_clean_ids.extend(np.random.choice(cls_ids, int(max_load_num), replace=False))
        else:
            balance_filter_clean_ids.extend(cls_ids)
    filter_clean_dataset = torch.utils.data.Subset(new_train_dataset, balance_filter_clean_ids)
    filter_clean_loader = torch.utils.data.DataLoader(DatasetWithIndices(filter_clean_dataset), batch_size=batch_size, shuffle=True)

    # check noise rate
    # hard_clean_ids = np.array(list(set(filter_clean_ids) | set(hard_relabel_ids)))
    if train_clean_labels is not None and train_hard_ids is not None:
        noise_flag = np.where(np.argmax(new_train_labels[new_hard_trainable_ids], axis=1) != train_clean_labels[new_hard_trainable_ids])[0]
        print("hard Relabel+clean noise rate",len(noise_flag)/len(new_hard_trainable_ids), len(new_hard_trainable_ids), len(noise_flag))
        print("Hard train sample num:", len(set(train_hard_ids) & set(new_hard_trainable_ids)))

        noise_flag = np.where(np.argmax(new_train_labels[balance_softnhard_relabel_ids], axis=1) != train_clean_labels[balance_softnhard_relabel_ids])[0]
        print("hard and soft Relabel+clean noise rate",len(noise_flag)/len(train_clean_labels), len(balance_softnhard_relabel_ids), len(noise_flag))
        print("Hard train sample num:", len(set(train_hard_ids) & set(balance_softnhard_relabel_ids)))
        
        noise_flag = np.where(np.argmax(new_train_labels[balance_filter_clean_ids], axis=1) != train_clean_labels[balance_filter_clean_ids])[0]
        print("clean noise rate", len(noise_flag)/len(balance_filter_clean_ids), len(balance_filter_clean_ids), len(noise_flag))
        print("Hard train sample num:", len(set(train_hard_ids) & set(balance_filter_clean_ids)))
    return softnhard_relabel_train_loader, hard_relabel_train_loader, filter_clean_loader, new_hard_trainable_ids, balance_softnhard_relabel_ids, balance_filter_clean_ids, new_train_labels




def get_sorted_checkpoints(directory, file_extension=".pth"):
    """
    Get a sorted list of checkpoint files in a directory based on the numerical value in the filename,
    ignoring files without numbers.

    Args:
        directory (str): Path to the directory containing the checkpoint files.
        file_extension (str): File extension to filter files (default: ".pth").

    Returns:
        list: A sorted list of checkpoint file paths with numbers in their names.
    """
    def extract_number(filename):
        """Extract the first number found in the filename."""
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else None  # Return None if no number is found

    # List all files with the specified extension in the directory
    checkpoint_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith(file_extension)
    ]

    # Filter files with numbers and sort them based on the extracted number
    numbered_files = [
        f for f in checkpoint_files if extract_number(os.path.basename(f)) is not None
    ]
    sorted_checkpoints = sorted(numbered_files, key=lambda f: extract_number(os.path.basename(f)))

    return sorted_checkpoints



def get_proxy_from_aum(aum_save_dir, labels, noise_ids=None, train_hard_ids=None, aum_threshold=1, proxy_num_percls=100, num_classes=10):
    import pandas as pd
    train_df = pd.read_csv(os.path.join(aum_save_dir, "aum_values.csv"))
    pred_ids = train_df['sample_id'][train_df['aum'] > aum_threshold].values
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    pred_labels = labels[pred_ids]
    pred_ids2cls = {}
    min_num = 1e7
    print("pred_ids:", len(pred_ids))
    print(pred_labels)
    
    for i in range(num_classes):
        pred_ids2cls[i] = pred_ids[np.where(pred_labels == i)[0]]
        # print("Class:", i, "Num:", len(pred_ids2cls[i]))
        if len(pred_ids2cls[i]) < min_num:
            min_num = len(pred_ids2cls[i])
    print("min_num:", min_num)

    if min_num > proxy_num_percls:
        min_num = proxy_num_percls
    # equal sample from each class
    pred_ids = []
    for i in range(num_classes):
        pred_ids.extend(pred_ids2cls[i][:min_num])
    print("balance pred_ids:", len(pred_ids))
    
    topk_aum_ids = pred_ids

    topk_labels = labels[topk_aum_ids]
    # count class distribution
    class_dist = np.bincount(topk_labels)
    print("topk class dist:", class_dist)
    print("total noise:", np.sum(np.isin(topk_aum_ids, noise_ids)))

    topk_noise_ids = np.where(np.isin(topk_aum_ids, noise_ids))[0]
    class_dist = np.bincount(topk_labels[topk_noise_ids])
    print("noise dist:", class_dist)
    topk_hard_ids = np.where(np.isin(topk_aum_ids, train_hard_ids))[0]
    class_dist = np.bincount(topk_labels[topk_hard_ids])
    print("hard dist:", class_dist, len(topk_hard_ids))


    return pred_ids
