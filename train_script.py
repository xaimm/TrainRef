import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from timm.models import create_model
import vision_transformer as vits

import modeling_pretrain
from utils import *


# ------------------------------
#   1) Model & Processor Setup
# ------------------------------

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

def load_model_and_transform(model_type, ckpt_path):
    """
    Loads the requested model and returns both:
      - model (nn.Module), set to eval mode
      - transform_fn (callable) to convert a PIL image into the correct input for the model

    Note: For the Hugging Face-based models (DINOv2, CLIP, EsViT),
    we wrap the HF processor in a function that returns a torch.Tensor.
    For SwAV (from TIMM), we use a standard torchvision transform pipeline.
    """
    model_type = model_type.lower()

    if model_type == "dinov2":
        from transformers import Dinov2Model, AutoImageProcessor
        model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        def transform_fn(pil_img: Image.Image):
            # Convert PIL Image -> pixel_values: [3, H, W]
            inputs = processor(images=pil_img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)
        
    elif "dino_" in model_type:
        
        arch = "vit_small"
        arch = "resnet50"
        arch = model_type.split("_")[1]
        checkpoint_key = 'teacher'
        
        if arch == "vit_small":
            pretrained_weights = '/home/mmr/code/SSL/dino/save/checkpoint0400.pth'
            # pretrained_weights = '/home/mmr/code/SSL/dino/save_6k/checkpoint0280.pth'
            patch_size = 16
            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            print(f"Model {arch} {patch_size}x{patch_size} built.")
        elif arch == "resnet50":
            import torchvision.models as torchvision_models
            import torch.nn as nn
            pretrained_weights = '/home/mmr/code/SSL/dino/save/dino_resnet50/checkpoint0260.pth'
            model = torchvision_models.__dict__[arch](num_classes=0)
            model.fc = nn.Identity()
            patch_size = -1
        
        model.cuda()
        load_pretrained_weights(model, pretrained_weights, checkpoint_key, arch, patch_size)
        
        transform_compose = transforms.Compose([
            # transforms.Resize(256, interpolation=3),
            # transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        def transform_fn(pil_img: Image.Image):
            # Convert PIL Image -> pixel_values: [3, H, W]
            return transform_compose(pil_img)
        
    elif 'beitv2' in model_type:
        import torch
        # Load Pre-trained Model
        model = create_model(
            'beit_base_patch16_224_8k_vocab_cls_pt',
            pretrained=False,
            drop_path_rate=0,
            drop_block_rate=None,
            use_shared_rel_pos_bias=True,
            use_abs_pos_emb=False,
            init_values=0.1,
            vocab_size=8192,
            early_layers=9,
            head_layers=2,
            shared_lm_head=True,
        )
        # print(torch)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # checkpoint = torch.load('/home/murong/code/SSL/unilm/beit2/save/beit_base_patch16_224_8k_vocab_webvision_pretrain/checkpoint.pth', map_location='cpu')
        # checkpoint = torch.load('/home/mmr/code/SSL/unilm/beit2/save/constrastive_finetuned_subtest/checkpoint.pth', map_location='cpu') # pos0.2 contrastive finetuning using influence ce clean
        # checkpoint = torch.load('/home/mmr/code/SSL/unilm/beit2/save/constrastive_finetuned_subtest/checkpoint.pth', map_location='cpu') # pos0.5 contrastive finetuning using influence ce clean
        # checkpoint = torch.load('/home/mmr/code/SSL/unilm/beit2/save/constrastive_finetuned_beitv2_pos05/checkpoint.pth', map_location='cpu')
        # checkpoint = torch.load('/home/mmr/code/XAI/UDA-pytorch_mod/temp_model/beitv2_finetune.pth', map_location='cpu')
        checkpoint = torch.load(ckpt_path, map_location='cpu')  
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        # Remove classification head (if necessary)
        model.head = torch.nn.Identity()

        # def extract_features_beitv2(model, dataloader, device='cuda'):
        #     features = []
        #     with torch.no_grad():
        #         for images, _ in tqdm(dataloader, total=len(dataloader)):  # Assuming dataloader returns (image, label)
        #             images = images.to(device)
        #             bool_masked_pos = torch.zeros((images.shape[0], 196), dtype=torch.bool).to(images.device)
        #             embeddings, _ = model.forward_features(images, bool_masked_pos)  # Extract features
        #             cls_token_embedding = embeddings[:, 0]  # Global embedding
        #             # print(cls_token_embedding)
        #             features.append(cls_token_embedding.cpu())
        #     return torch.cat(features, dim=0)
        
        transform_compose=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        def transform_fn(pil_img: Image.Image):
            return transform_compose(pil_img)
        
        
    elif model_type == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def transform_fn(pil_img: Image.Image):
            inputs = processor(images=pil_img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

    elif model_type == "swav":
        import torch
        print("Loading SwAV ResNet-50 from facebookresearch/swav:main via Torch Hub...")
        model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        # Turn the final classifier (fc) into an Identity layer
        model.fc = torch.nn.Identity()
        model.eval()

        # Typical ImageNet transforms
        transform_compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        def transform_fn(pil_img: Image.Image):
            return transform_compose(pil_img)

    elif model_type == "esvit":
        from transformers import EsvitModel, AutoImageProcessor
        model = EsvitModel.from_pretrained("microsoft/esvit-base-p16")
        processor = AutoImageProcessor.from_pretrained("microsoft/esvit-base-p16")

        def transform_fn(pil_img: Image.Image):
            inputs = processor(images=pil_img, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model, transform_fn


def get_image_features(images: torch.Tensor, model: nn.Module, model_type: str):
    """
    Forward pass to extract features for each model type.
    """
    with torch.no_grad():
        if model_type == "dinov2":
            outputs = model(pixel_values=images)
            feats = outputs.pooler_output  # [batch_size, hidden_dim]
        elif 'dino_' in model_type:
            feats = model(images)
        elif 'beitv2' in model_type:
            bool_masked_pos = torch.zeros((images.shape[0], 196), dtype=torch.bool).to(images.device)
            embeddings, _ = model.forward_features(images, bool_masked_pos)  # Extract features
            feats = embeddings[:, 0]  # Global embedding
        elif model_type == "clip":
            feats = model.get_image_features(pixel_values=images)  # [batch_size, embed_dim]
        elif model_type == "swav":
            # Now that model.fc = Identity, 
            # model(images) returns [batch_size, 2048] (penultimate layer).
            feats = model(images)  # [batch_size, 2048]
        elif model_type == "esvit":
            outputs = model(pixel_values=images)
            feats = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return feats


# -----------------------------------
#   2) Original Dataset Definitions
# -----------------------------------

class animal10N_dataset(Dataset):
    """
    Animal-10N dataset from your original code.
    Accepts a root_dir ('/home/mmr/data/Animals-10N/raw_image_ver')
    and a 'mode' ('train' or 'test'), plus a transform function.
    """
    def __init__(self, root_dir, mode, transform_fn):
        self.root_dir = root_dir
        self.mode = mode
        self.transform_fn = transform_fn

        self.train_dir = os.path.join(root_dir, 'training')
        self.test_dir = os.path.join(root_dir, 'testing')
        train_imgs = os.listdir(self.train_dir)
        test_imgs = os.listdir(self.test_dir)

        self.train_imgs = []
        self.test_imgs = []
        self.train_labels = []
        self.test_labels = []
        for img in train_imgs:
            # label = int(img[0]) in your original code
            self.train_imgs.append([img, int(img[0])])
            self.train_labels.append(int(img[0]))
        for img in test_imgs:
            self.test_imgs.append([img, int(img[0])])
            self.test_labels.append(int(img[0]))

    def __getitem__(self, index):
        if self.mode == 'train':
            img_id, target = self.train_imgs[index]
            img_path = os.path.join(self.train_dir, img_id)
        else:  # 'test'
            img_id, target = self.test_imgs[index]
            img_path = os.path.join(self.test_dir, img_id)

        image = Image.open(img_path).convert('RGB')
        image = self.transform_fn(image)  # e.g. pixel_values or standard transforms
        return image, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_imgs)
        else:
            return len(self.test_imgs)


class webvision_dataset(Dataset):
    """
    WebVision dataset from your original code.
    """
    def __init__(self, root_dir, mode, num_class, transform_fn):
        self.root = root_dir
        self.mode = mode
        self.transform_fn = transform_fn

        if self.mode == 'test':
            self.val_imgs = []
            self.val_labels = []
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target < num_class:
                        self.val_imgs.append(img)
                        self.val_labels.append(target)
        else:
            self.train_imgs = []
            self.train_labels = []
            with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target < num_class:
                        self.train_imgs.append(img)
                        self.train_labels.append(target)

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            full_path = os.path.join(self.root, img_path)
        else:  # 'test'
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
            full_path = os.path.join(self.root, 'val_images_256', img_path)

        image = Image.open(full_path).convert('RGB')
        image = self.transform_fn(image)
        return image, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)

# For CIFAR-10, we can just use torchvision.datasets.CIFAR10
# with a custom transform or processor. We'll handle that in main().

# --------------------------
#   3) Feature Extraction
# --------------------------

def extract_features(model, dataloader, device, model_type, save_path):
    """
    Extract features from a pretrained model and save them to disk.
    """
    model.to(device)
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device)
            feats = get_image_features(images, model, model_type)
            features.append(feats.cpu())
            labels.append(lbls)

    features = torch.cat(features, dim=0)  # [N, feat_dim]
    labels = torch.cat(labels, dim=0)      # [N]
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "features.npy"), features.numpy())
    np.save(os.path.join(save_path, "labels.npy"), labels.numpy())
    print(f"Features and labels saved to {save_path}")

# Define the MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=1, hidden_dim=512, dropout=0):
        """
        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            num_layers (int): Number of layers (including output layer).
            hidden_dim (int): Number of units in each hidden layer.
            dropout (float): Dropout probability.
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            # Add the first hidden layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Add intermediate hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            # Add the output layer
            layers.append(nn.Linear(hidden_dim, num_classes))
        
        
            
        # Register hooks
        # self.fc.register_forward_hook(self.forward_hook)
        # self.fc.register_full_backward_hook(self.backward_hook)
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(self.forward_hook)
                layer.register_backward_hook(self.backward_hook)
        

        
        self.classifier = nn.Sequential(*layers)
        
        # self.classifier = nn.Linear(input_dim, num_classes)

    def forward_hook(self, module, input, output):
        module.activations = input[0]

    def backward_hook(self, module, grad_input, grad_output):
        if len(grad_output[0].shape) > 2:
            averaged_grad_output = torch.mean(grad_output[0], dim=(2, 3))
            averaged_activations = torch.mean(module.activations, dim=(2, 3))
        else:
            averaged_grad_output = grad_output[0]
            averaged_activations = module.activations
        module.grad_sample = torch.einsum('n...i,n...j->nij', averaged_grad_output, averaged_activations)
        
    def get_persample_grad(self):
        
        # fc_grad = self.fc.grad_sample
        # b_s = fc_grad.shape[0]
        # fc_grad = fc_grad.view(b_s, -1).detach().clone()
        grads = []
        for layer in self.classifier:
            if hasattr(layer, 'grad_sample'):
                layer_grad = layer.grad_sample
                b_s = layer_grad.shape[0]
                layer_grad = layer_grad.view(b_s, -1).detach().clone()
                grads.append(layer_grad)
        grads = torch.cat(grads, dim=1)
        return grads

    def forward(self, x):
        return self.classifier(x)

def test_mlp_model(classifier, test_loader, device):
    """
    Test the MLP model on the test dataset.
    """
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feature_batch, label_batch in tqdm(test_loader, desc="Testing"):
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            outputs = classifier(feature_batch)
            _, predicted = torch.max(outputs, dim=1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# ------------------------
#       Main Example
# ------------------------
if __name__ == "__main__":
    """
    Example usage:
      python script.py
    Then pick a model_type among ["dinov2", "clip", "swav", "esvit"]
    and a dataset_name among ["cifar10", "animal10N", "webvision"].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose your model
    # model_type = "dinov2"  # Options: "dinov2", "clip", "swav", "esvit"
    model_type = "beitv2_webvision"
    # Choose your dataset
    dataset_name = "webvision"  # Options: "cifar10", "animal10N", "webvision"
    task_name = "webvision_inpretrain_infexpand"
    # Hyperparameters, etc.
    batch_size = 64
    num_workers = 1
    feature_dir = "./features"

    # -------------------------
    #  Load model + transform
    # -------------------------
    ckpt_path = '/home/murong/code/SSL/unilm/beit2/save/beitv2_base_patch16_224_pt1k/beitv2_base_patch16_224_pt1k.pth'
    model, transform_fn = load_model_and_transform(model_type, ckpt_path)

    # -------------------------
    #   Dataset + DataLoader
    # -------------------------
    if dataset_name == "animal10N":
        # Animal-10N
        num_classes = 10
        root_dir = "/home/murong/data/Animals-10N/raw_image_ver"  # adjust if needed
        classes = [
            "cat",
            "lynx",
            "wolf",
            "coyote",
            "cheetah",
            "jaguar",
            "chimpanzee",
            "orangutan",
            "hamster",
            "guinea pig"
        ]

        img_train_dataset = animal10N_dataset(root_dir=root_dir, mode='train', transform_fn=transform_fn)
        img_test_dataset  = animal10N_dataset(root_dir=root_dir, mode='test',  transform_fn=transform_fn)

        train_loader = DataLoader(img_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(img_test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset_name == "webvision":
        # WebVision
        root_dir = "/home/murong/data/webvision/"  # adjust if needed
        num_classes = 50
        num_workers = 4
        data_dir = './data'
        feature_dir = './features'
        batch_size = 256
        syn_map_file = '/home/murong/data/webvision/info/queries_synsets_map.txt'
        queries_file = '/home/murong/data/webvision/info/queries_google.txt'

        queries_dict = {}
        with open(queries_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                queries_dict[int(line[0])] = line[1]

        syn_dict = {}
        with open(syn_map_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                n_query, n_label = int(line.strip().split()[0]), int(line.strip().split()[1])-1
                if n_label not in syn_dict:
                    syn_dict[n_label] = []
                syn_dict[n_label].append(queries_dict[n_query])
        classes = []
        for i in range(num_classes):
            classes.append(syn_dict[i][0])

        img_train_dataset = webvision_dataset(root_dir, 'train', num_classes, transform_fn)
        img_test_dataset  = webvision_dataset(root_dir, 'test', num_classes, transform_fn)

        train_loader = DataLoader(img_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(img_test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset_name == "cifar10":
        # CIFAR-10 (from torchvision)
        # We need a custom transform function. For HF-based models, the transform is the HF processor.
        # For timm-based models (SwAV), we have a standard Compose. We'll apply that transform directly
        # in the dataset: but CIFAR10 returns PIL by default, which we can pass to transform_fn.

        num_classes = 10
        root_dir = "~/data"

        # The official torchvision CIFAR10 dataset
        # We'll transform each image on-the-fly using transform_fn in the collate_fn:
        train_dataset_cifar = datasets.CIFAR10(root=root_dir, train=True,  download=True)
        test_dataset_cifar  = datasets.CIFAR10(root=root_dir, train=False, download=True)

        # We define a collate_fn that uses transform_fn on each PIL image
        def collate_fn(batch):
            imgs, labels = zip(*batch)
            tensors = []
            for img in imgs:
                # Each img is a PIL Image; apply the transform
                tensors.append(transform_fn(img))
            # Stack into a batch of shape [batch_size, 3, 224, 224], for instance
            return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long)

        train_loader = DataLoader(train_dataset_cifar, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader  = DataLoader(test_dataset_cifar,  batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
save_path_train = os.path.join(feature_dir, dataset_name, model_type, f"train_{task_name}")
save_path_test  = os.path.join(feature_dir, dataset_name, model_type, f"test_{task_name}")


# -------------------------
#   Extract + Save Features
# -------------------------


extract_features(model, train_loader, device, model_type, save_path_train)
extract_features(model, test_loader,  device, model_type, save_path_test)



# Load features and labels
train_features = torch.tensor(np.load(os.path.join(save_path_train, "features.npy")))
train_labels = torch.tensor(np.load(os.path.join(save_path_train, "labels.npy")))

# Load features and labels
test_features = torch.tensor(np.load(os.path.join(save_path_test, "features.npy")))
test_labels = torch.tensor(np.load(os.path.join(save_path_test, "labels.npy")))


combine_feature = False
if combine_feature:
    dinov2_train_features = torch.tensor(np.load(os.path.join(feature_dir, 'webvision', 'dinov2', 'train', 'features.npy')))
    clip_train_features = torch.tensor(np.load(os.path.join(feature_dir, 'webvision', 'clip', 'train', 'features.npy')))
    train_features = torch.cat((dinov2_train_features, clip_train_features), dim=1)
    train_labels = torch.tensor(np.load(os.path.join(feature_dir, 'webvision', 'dinov2', 'train', 'labels.npy')))
    
    dinov2_test_features = torch.tensor(np.load(os.path.join(feature_dir, 'webvision', 'dinov2', 'test', 'features.npy')))
    clip_test_features = torch.tensor(np.load(os.path.join(feature_dir, 'webvision', 'clip', 'test', 'features.npy')))
    test_features = torch.cat((dinov2_test_features, clip_test_features), dim=1)
    test_labels = torch.tensor(np.load(os.path.join(feature_dir, 'webvision', 'dinov2', 'test', 'labels.npy')))


print("train feature shape:", train_features.shape)
print("test feature shape:", test_features.shape)

# load selection
selection_flag = False
if selection_flag:
    selection_method = "disc_selection"
    # selection_method = "inf_selection_ent"
    # clean_ids = np.load(f"/home/mmr/code/XAI/DISC_inf/selection/webvision/{selection_method}/filter_clean_ids.npy")
    # hard_ids = np.load(f"/home/mmr/code/XAI/DISC_inf/selection/webvision/{selection_method}/filter_hard_ids.npy")
    # trainable_ids = np.concatenate((clean_ids, hard_ids))
    # train_features = train_features[trainable_ids]
    # train_labels = train_labels[trainable_ids]
    
    # clean_ids = np.load(f'/home/mmr/code/XAI/UDA-pytorch_mod/softlabel/selection/dinov2/webvision/emb/filter_clean_ids.npy')
    
    clean_ids = np.load('/home/mmr/code/XAI/UDA-pytorch_mod/softlabel/ood_conft/selection/webvision/emb/filter_train_ids.npy')
    clean_labels = np.load('/home/mmr/code/XAI/UDA-pytorch_mod/softlabel/ood_conft/selection/webvision/emb/filter_train_labels.npy')
    
    train_features = train_features[clean_ids]
    train_labels = train_labels[clean_ids]
    train_labels = torch.tensor(clean_labels)
    # train_labels = None


# Prepare MLP training data
train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("train feature shape:", train_features.shape)
print("test feature shape:", test_features.shape)

# Define and train the MLP classifier
input_dim = train_features.size(1)
num_classes = num_classes
classifier = MLPClassifier(input_dim, num_classes, num_layers=1).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(classifier.parameters(), lr=1e-5)
optimizer = optim.SGD(classifier.parameters(), lr=1e-3, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0], gamma=1)
num_epochs = 30
save_interval = 10
ckpt_save_dir = f"./softlabel/{dataset_name+model_type}/MLP/checkpoints"
if not os.path.exists(ckpt_save_dir):
    os.makedirs(ckpt_save_dir)


for epoch in range(num_epochs):
    classifier.train()
    epoch_loss = 0
    total = 0
    correct = 0
    for feature_batch, label_batch in train_loader:
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)

        optimizer.zero_grad()
        outputs = classifier(feature_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        total += label_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        # _, label_batch = torch.max(label_batch, 1)
        correct += (predicted == label_batch).sum().item()
    scheduler.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    test_mlp_model(classifier, test_loader, device)
    if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
        state_dict = {}
        state_dict['model'] = classifier.state_dict()
        state_dict['optimizer'] = optimizer.state_dict()
        torch.save(state_dict, os.path.join(ckpt_save_dir, f"checkpoint_{epoch + 1}.pt"))

print("Training complete!")

test_mlp_model(classifier, test_loader, device)

# if __name__ == "__main__":
#     main()

ckpt_list = get_sorted_checkpoints(ckpt_save_dir, ".pt")[:3]
print(ckpt_list)


sub_test_indices = np.random.choice(len(test_loader.dataset), 500, replace=False)
# sub_test_indices = np.arange(len(test_loader.dataset))

sub_test_dataset = torch.utils.data.Subset(test_loader.dataset, sub_test_indices)
val_loader = DataLoader(sub_test_dataset, batch_size=1000, shuffle=False)

# clean_ids = np.load('/home/mmr/code/XAI/UDA-pytorch_mod/save/webvison/inf_selection_ce/filter_clean_ids.npy')
# # equally sample 200 samples from each class in clean_ids
# train_labels = np.array([img_train_dataset.train_labels[img_train_dataset.train_imgs[i]] for i in range(len(img_train_dataset.train_imgs))])
# balanced_clean_ids = []
# for i in range(num_classes):
#     class_ids = clean_ids[np.where(train_labels[clean_ids] == i)[0]]
#     balanced_clean_ids.extend(np.random.choice(class_ids, 200, replace=False))
# balanced_clean_ids = np.array(balanced_clean_ids)
# sub_train_dataset = torch.utils.data.Subset(train_loader.dataset, balanced_clean_ids)
# val_loader = DataLoader(sub_train_dataset, batch_size=1000, shuffle=False)


target_train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)

inf_cls, allone_inf_cls, inf_scores_sum, allone_inf_scores_sum, emb_cossim_sum, allone_grad_cossim_sum = get_influence_scores(classifier, target_train_loader, val_loader, criterion, ckpt_list, num_classes=num_classes)


def get_voting_results_optimized(
    insp_ids, 
    emb_cossim_sum, 
    allone_inf_cls, 
    img_test_dataset, 
    sub_test_indices, 
    num_classes, 
    similarity_threshold=0.7
):
    """
    Optimized function to compute voting results based on embedding cosine similarities and inference classes.
    
    Parameters:
    - insp_ids (array-like): Indices of inspections to process.
    - emb_cossim_sum (np.ndarray): 2D array of cosine similarity sums (shape: [num_inspections, num_embeddings]).
    - inf_cls (np.ndarray): 2D array of inference class scores (shape: [num_inspections, num_classes]).
    - img_test_dataset (list or array-like): Dataset containing image information; each element should have a class label at index 1.
    - sub_test_indices (array-like): Indices to subset `img_test_dataset`.
    - num_classes (int): Total number of classes.
    - similarity_threshold (float): Threshold for cosine similarity to consider in voting.
    
    Returns:
    - emb_vote_results (np.ndarray): Normalized embedding-based voting results (shape: [len(insp_ids), num_classes]).
    - inf_vote_results (np.ndarray): Normalized inference-based voting results (shape: [len(insp_ids), num_classes]).
    """
    
    # Precompute class labels for sub_test_indices
    # Assuming that img_test_dataset[i][1] contains the class label as an integer
    classes = np.array([img_test_dataset[i][1] for i in sub_test_indices], dtype=np.int32)
    
    # Extract the relevant subset of emb_cossim_sum
    # Shape: (N, M) where N = len(insp_ids), M = num_embeddings
    emb_cossim_subset = emb_cossim_sum[insp_ids]
    
    # Vectorized Inference Voting
    # Normalize inference class scores for all insp_ids at once
    # Shape: (N, num_classes)
    inf_vote_subset = allone_inf_cls[insp_ids]
    inf_vote_subset = -inf_vote_subset  # Invert scores to align with cosine similarity
    inf_vote_subset = np.maximum(inf_vote_subset, 0)  # Ensure non-negative values
    inf_vote_sums = inf_vote_subset.sum(axis=1, keepdims=True)  # Shape: (N, 1)
    
    # To avoid division by zero, set zero sums to one (resulting in zero vectors after division)
    inf_vote_sums[inf_vote_sums == 0] = 1.0
    inf_vote_results = inf_vote_subset / inf_vote_sums  # Shape: (N, num_classes)
    
    # Initialize embedding vote results
    emb_vote_results = np.zeros((len(insp_ids), num_classes), dtype=np.float64)
    
    # Iterate over each inspection ID to compute embedding-based voting
    for idx, emb_cossim in enumerate(tqdm(emb_cossim_subset, desc="Processing Embedding Votes")):
        # Identify embeddings with similarity above the threshold
        vote_mask = emb_cossim > similarity_threshold
        emb_vote_num = np.count_nonzero(vote_mask)
        emb_vote_num =10
        
        if emb_vote_num < 5:
            # If no embeddings exceed the threshold, skip to avoid division by zero
            continue
        
        # Get the indices of embeddings sorted by descending similarity
        sorted_indices = np.argsort(-emb_cossim)
        
        # Select top embeddings that exceed the similarity threshold
        top_vote_ids = sorted_indices[:emb_vote_num]
        
        # Retrieve the corresponding classes and similarities
        vote_classes = classes[top_vote_ids]
        vote_similarities = emb_cossim[top_vote_ids]
        
        # Accumulate vote sums per class using np.bincount
        # Ensure that vote_classes are non-negative integers less than num_classes
        vote_sum = np.bincount(vote_classes, weights=vote_similarities, minlength=num_classes)
        
        # Normalize the vote sums to obtain probabilities
        vote_total = vote_sum.sum()
        if vote_total > 0:
            emb_vote_results[idx] = vote_sum / vote_total
        else:
            # If vote_total is zero, leave the vote_result as zeros
            pass
    
    return emb_vote_results, inf_vote_results

emb_vote_results, inf_vote_results = get_voting_results_optimized(
    np.array(range(len(train_dataset))),
    emb_cossim_sum,
    allone_inf_cls,
    img_test_dataset,
    sub_test_indices,
    num_classes
)

assigned_labels = np.array([label for _, label in train_loader.dataset])
assigned_onehot_labels = np.eye(num_classes)[assigned_labels]
# compute the cross entropy between emb_vote_results and assigned labels
def cross_entropy(p, q):
    return -np.sum(p * np.log(q + 1e-9), axis=1)  # Adding epsilon to avoid log(0)

nega_inf_cls = -inf_cls * (inf_cls < -0.1)
nega_inf_sum = np.sum(nega_inf_cls, axis=1)

emb_assigned = emb_vote_results[np.arange(len(assigned_labels)), assigned_labels]
inf_assigned = inf_vote_results[np.arange(len(assigned_labels)), assigned_labels]
emb_assigned = np.array(emb_assigned)

emb_ce = cross_entropy(assigned_onehot_labels, emb_vote_results)
inf_ce = cross_entropy(assigned_onehot_labels, inf_vote_results)


emb_threshold = 0.1
inf_threshold = 0.5

emb_noise_ids = np.where(emb_ce > emb_threshold)[0]
emb_clean_ids = np.where(emb_ce <= emb_threshold)[0]
inf_noise_ids = np.where(inf_ce > inf_threshold)[0]
inf_clean_ids = np.where(inf_ce <= inf_threshold)[0]

emb_noise_ids = np.where(emb_assigned < emb_threshold)[0]
emb_clean_ids = np.where(emb_assigned >= emb_threshold)[0]

inf_noise_ids = np.where(inf_assigned < inf_threshold)[0]
inf_clean_ids = np.where(inf_assigned >= inf_threshold)[0]

# inf_noise_ids = np.where(nega_inf_sum > inf_threshold)[0]
# inf_clean_ids = np.where(nega_inf_sum <= inf_threshold)[0]

print("emb detection num:", len(emb_noise_ids), "inf detection num", len(inf_noise_ids))

disc_clean_ids = np.load('/home/murong/code/XAI/DISC_inf/selection/webvision/disc_selection/filter_clean_ids.npy')
disc_hard_ids = np.load('/home/murong/code/XAI/DISC_inf/selection/webvision/disc_selection/filter_hard_ids.npy')
disc_noise_ids = np.array(list(set(range(len(train_dataset))) - set(disc_clean_ids) - set(disc_hard_ids)))

print("disc detection num:", len(disc_noise_ids))

# check intersection between emb_noise_ids and disc_noise_ids
print("emb disc intersection:", len(set(emb_noise_ids) & set(disc_noise_ids)))
print("inf disc intersection:", len(set(inf_noise_ids) & set(disc_noise_ids)))

only_emb_noise_ids = np.array(list(set(emb_noise_ids) - set(disc_noise_ids)))
only_inf_noise_ids = np.array(list(set(inf_noise_ids) - set(disc_noise_ids)))

only_disc_noise_ids_emb = np.array(list(set(disc_noise_ids) - set(emb_noise_ids)))
only_disc_noise_ids_inf = np.array(list(set(disc_noise_ids) - set(inf_noise_ids)))

emb_inf_clean_ids = np.array(list(set(emb_clean_ids) & set(inf_clean_ids)))
print("Emb clean num:", len(emb_clean_ids))
print("inf clean num:", len(inf_clean_ids))
print("Emb Inf clean num:", len(emb_inf_clean_ids))

# detect ood according to emb_cos_sim
emb_cossim_sum.shape

from scipy.spatial.distance import mahalanobis
import numpy as np
from scipy.linalg import inv


# clean_ids = np.load('/home/mmr/code/XAI/UDA-pytorch_mod/save/webvison/inf_selection_ce/filter_clean_ids.npy')
clean_ids = inf_clean_ids
# equally sample 200 samples from each class in clean_ids
train_labels = np.array(img_train_dataset.train_labels)
balanced_clean_ids = []
for i in range(num_classes):
    class_ids = clean_ids[np.where(train_labels[clean_ids] == i)[0]]
    balanced_clean_ids.extend(np.random.choice(class_ids, 100, replace=False))
balanced_clean_ids = np.array(balanced_clean_ids)


# Compute mean and covariance matrix of ID data
val_features = train_features[inf_clean_ids]
# val_features = test_features[sub_test_indices]

mean_vec = np.mean(val_features.numpy(), axis=0)
cov_matrix = np.cov(val_features.numpy(), rowvar=False)
cov_inv = inv(cov_matrix)

print(val_features.shape, mean_vec.shape, cov_matrix.shape)
# print(mean_vec, cov_matrix)

# Mahalanobis distance for a test sample
# Loop through each sample in train_features
distances = [
    mahalanobis(sample, mean_vec, cov_inv)
    for sample in train_features.numpy()
]

# Convert distances to a numpy array for further analysis
distances = np.array(distances)


# Compute Mahalanobis distances for ID samples
distances_id = [
    mahalanobis(sample, mean_vec, cov_inv)
    for sample in test_features[sub_test_indices].numpy()
]

# Set the threshold as the 95th percentile of ID distances
threshold_95 = np.percentile(distances_id, 95)
print("Percentile-based Threshold (95):", threshold_95)

dist_rank = np.argsort(distances)[::-1]
# ood_ids = dist_rank[:5000]
# Define threshold (using validation data or statistical methods)
# threshold = 55  # Set threshold
is_ood = distances > threshold_95
ood_ids = np.where(is_ood)[0]

# how many ood in disc_noise_ids
print("ood num:", len(ood_ids))
print("ood disc intersection:", len(set(ood_ids) & set(disc_noise_ids)))
print("ood emb intersection:", len(set(ood_ids) & set(emb_noise_ids)))
print("ood inf intersection:", len(set(ood_ids) & set(inf_noise_ids)))


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def compute_knn_distances(feature_matrix, id_data, k=5):
    """
    Compute the k-NN distance for each sample in the feature_matrix using ID data as the reference.
    The distance is computed based on cosine similarity.

    Parameters:
    - feature_matrix: np.ndarray of shape (N, dim), data under estimation.
    - id_data: np.ndarray of shape (m, dim), in-distribution (ID) data.
    - k: int, number of nearest neighbors to consider.

    Returns:
    - knn_distances: np.ndarray of shape (N,), average k-NN distances for each sample in feature_matrix.
    """
    # Compute pairwise cosine similarity between feature_matrix and id_data
    similarity = cosine_similarity(feature_matrix, id_data)

    # Convert similarity to distance (1 - similarity)
    distances = 1 - similarity

    # Find the k nearest neighbors for each sample
    # Sort distances along the last axis
    sorted_distances = np.sort(distances, axis=1)

    # Take the mean of the top-k nearest neighbors
    knn_distances = np.mean(sorted_distances[:, :k], axis=1)

    return knn_distances



knn_distances = compute_knn_distances(train_features, test_features[sub_test_indices], k=10)

# plot distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
plt.hist(knn_distances, bins=100)
plt.show()


proxy_knn_distances = compute_knn_distances(test_features, test_features[sub_test_indices], k=10)
print("range of knn distances:", np.min(proxy_knn_distances), np.max(proxy_knn_distances))

threshold_95 = np.percentile(proxy_knn_distances, 95)
print("Percentile-based Threshold (95):", threshold_95)

near_ids = np.where(knn_distances < threshold_95)[0]
far_ids = np.where(knn_distances >= threshold_95)[0]

print("near num:", len(near_ids), "far num:", len(far_ids))

near_wo_ood_ids = np.setdiff1d(near_ids, ood_ids)
far_wo_ood_ids = np.setdiff1d(far_ids, ood_ids)
print("near without ood num:", len(near_wo_ood_ids), "far without ood num:", len(far_wo_ood_ids))


near_inf_noise_ids = np.intersect1d(near_wo_ood_ids, inf_noise_ids)
far_inf_noise_ids = np.intersect1d(far_ids, inf_noise_ids)
print("near inf noise num:", len(near_inf_noise_ids), "far inf noise num:", len(far_inf_noise_ids))

inf_clean_wo_ood_ids = np.setdiff1d(inf_clean_ids, ood_ids)
print("inf clean without ood num:", len(inf_clean_wo_ood_ids))


final_train_ids = np.concatenate([inf_clean_wo_ood_ids, near_inf_noise_ids])
train_onehot_targets = np.eye(num_classes)[np.array(img_train_dataset.train_labels)]
final_train_labels = np.concatenate([train_onehot_targets[inf_clean_wo_ood_ids], inf_vote_results[near_inf_noise_ids]])
print(len(final_train_ids), final_train_labels.shape)

# save

selection_save_dir = f'./softlabel/ood_conft/selection/webvision/{task_name}/'

if not os.path.exists(selection_save_dir):
    os.makedirs(selection_save_dir)
# save emb clean ids
np.save(selection_save_dir + 'filter_clean_ids.npy', inf_clean_wo_ood_ids)
np.save(selection_save_dir + 'filter_train_ids.npy', final_train_ids)
np.save(selection_save_dir + 'filter_train_labels.npy', final_train_labels)