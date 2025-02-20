import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



class DatasetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return (data, target), index

    def __len__(self):
        return len(self.dataset)


def get_influence_scores_batch(model, train_emb_batch, val_emb_batch, criterion, device='cuda', num_classes=10, lr=None):
    
    lr = lr if lr is not None else 0.01
    train_emb, train_target, allone_train_target = train_emb_batch[0].to(device), train_emb_batch[1].to(device), torch.ones((train_emb_batch[1].shape[0], num_classes)).to(device)
    val_emb, val_target = val_emb_batch[0].to(device), val_emb_batch[1].to(device)
    
    import copy
    # get last layer
    lastlayer = copy.deepcopy(model)
    lastlayer = lastlayer.to(device)
    optimizer = torch.optim.SGD(lastlayer.parameters(), lr=lr)
    optimizer.zero_grad()
    train_output = lastlayer(train_emb.detach())
    # print(train_emb.shape, train_target.shape)
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
    lastlayer = copy.deepcopy(model)
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
    lastlayer = copy.deepcopy(model)
    lastlayer = lastlayer.to(device)
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
    
    # graddot = (train_grad @ val_grad.T).detach().cpu().numpy()
    # embdot = (train_emb @ val_emb.T).detach().cpu().numpy()
    # allone_graddot = (allone_train_grad @ val_grad.T).detach().cpu().numpy()
    
    # influence_scores = embdot * graddot
    # allone_influence_scores = embdot * allone_graddot
    
    # emb_cossim = embdot
    # allone_grad_cossim = allone_graddot
    # grad_cossim = graddot
    
    
    # free memory
    # del train_output, train_loss, train_grad, val_output, val_loss, val_grad, allone_train_grad
    # torch.cuda.empty_cache()
    
    return influence_scores, allone_influence_scores, emb_cossim, allone_grad_cossim
    



def get_influence_scores(model, target_train_loader, val_loader, criterion, ckpt_list, num_classes=10):
    inf_list = []
    allone_inf_list = []
    emb_cossim_list = []
    allone_grad_cossim_list = []
    lr_list = []
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

        inf_scores = torch.zeros(len(target_train_loader.dataset), len(val_loader.dataset))
        allone_inf_scores = torch.zeros(len(target_train_loader.dataset), len(val_loader.dataset))
        emb_cossim = torch.zeros(len(target_train_loader.dataset), len(val_loader.dataset))
        allone_grad_cossim = torch.zeros(len(target_train_loader.dataset), len(val_loader.dataset))
        
        target_train_loader_ = DataLoader(DatasetWithIndices(target_train_loader.dataset), batch_size=target_train_loader.batch_size, shuffle=False)
        val_loader_ = DataLoader(DatasetWithIndices(val_loader.dataset), batch_size=val_loader.batch_size, shuffle=False)
        
        print("Computing influence scores...")
        progressbar = tqdm(total=len(val_loader_) * len(target_train_loader_))
        for val_emb_batch, val_indices in val_loader_:    
            for train_emb_batch, train_indices in target_train_loader_:
        # for train_emb_batch, train_indices in tqdm(target_train_loader):
        #     for val_emb_batch, val_indices in val_loader:
                # print(train_emb_batch)
                influence_scores_batch, allone_influence_scores_batch, emb_cossim_batch, allone_grad_cossim_batch = get_influence_scores_batch(model, train_emb_batch, val_emb_batch, criterion, num_classes=num_classes)
                inf_scores[train_indices[:, None], val_indices[None, :]] = torch.tensor(influence_scores_batch)
                allone_inf_scores[train_indices[:, None], val_indices[None, :]] = torch.tensor(allone_influence_scores_batch)
                emb_cossim[train_indices[:, None], val_indices[None, :]] = torch.tensor(emb_cossim_batch)
                allone_grad_cossim[train_indices[:, None], val_indices[None, :]] = torch.tensor(allone_grad_cossim_batch)
                # print("inf_scores:", inf_scores)
                # print("allone inf scores:", allone_inf_scores)
                # progress bar
                progressbar.update()
        progressbar.close()
                
        inf_list.append(inf_scores.numpy())
        allone_inf_list.append(allone_inf_scores.numpy())
        emb_cossim_list.append(emb_cossim.numpy())
        allone_grad_cossim_list.append(allone_grad_cossim.numpy())
        
    
    # weighted sum by lr_list 
    inf_scores_weighted_sum = np.zeros((len(target_train_loader.dataset), len(val_loader.dataset)))
    allone_inf_scores_weighted_sum = np.zeros((len(target_train_loader.dataset), len(val_loader.dataset)))
    emb_cossim_sum = np.zeros((len(target_train_loader.dataset), len(val_loader.dataset)))
    allone_grad_cossim_sum = np.zeros((len(target_train_loader.dataset), len(val_loader.dataset)))
    for i in range(len(ckpt_list)):
        inf_scores_weighted_sum += inf_list[i] * lr_list[i]
        allone_inf_scores_weighted_sum += allone_inf_list[i] * lr_list[i]
        emb_cossim_sum += emb_cossim_list[i] * lr_list[i]
        allone_grad_cossim_sum += allone_grad_cossim_list[i] * lr_list[i]
    
    inf_scores_sum = inf_scores_weighted_sum
    allone_inf_scores_sum = allone_inf_scores_weighted_sum
    
    inf_scores_sum = inf_scores_sum / len(ckpt_list)
    allone_inf_scores_sum = allone_inf_scores_sum / len(ckpt_list)
    emb_cossim_sum = emb_cossim_sum / len(ckpt_list)
    allone_grad_cossim_sum = allone_grad_cossim_sum / len(ckpt_list)
    
    # inf_scores_sum = np.sum(inf_list, axis=0)
    # print(inf_scores_sum.sum(axis=1))
    # print("inf_scores_sum", inf_scores_sum)

    val_targets = torch.zeros(len(val_loader.dataset), dtype=torch.long)
    for batch, val_indices in val_loader_:
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
    
    return inf_cls, allone_inf_cls, inf_scores_sum, allone_inf_scores_sum, emb_cossim_sum, allone_grad_cossim_sum



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
