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

# set gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def load_model_and_transform(model_type="dinov2"):
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
        
    elif "beit" in model_type:
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
        # checkpoint = torch.load('/home/murong/code/SSL/unilm/beit2/save/beit_base_patch16_224_8k_vocab_animal10n_pretrain_good/checkpoint.pth', map_location='cpu')
        # checkpoint = torch.load('/home/murong/code/SSL/unilm/beit2/save/constrastive_finetuned_clsbias_beitv2_animal10n_infexpand1_pos05/checkpoint.pth', map_location='cpu')
        # checkpoint = torch.load('/home/mmr/code/SSL/unilm/beit2/save/constrastive_finetuned_subtest/checkpoint.pth', map_location='cpu') # pos0.2 contrastive finetuning using influence ce clean
        # checkpoint = torch.load('/home/mmr/code/SSL/unilm/beit2/save/constrastive_finetuned_subtest/checkpoint.pth', map_location='cpu') # pos0.5 contrastive finetuning using influence ce clean
        # checkpoint = torch.load('/home/mmr/code/SSL/unilm/beit2/save/constrastive_finetuned_beitv2_pos05/checkpoint.pth', map_location='cpu')
        # checkpoint = torch.load('/home/mmr/code/XAI/UDA-pytorch_mod/temp_model/beitv2_finetune.pth', map_location='cpu')
        # checkpoint = torch.load('/home/murong/code/XAI/UDA-pytorch_mod/temp_model/webvision/beitv2_clsfinetune_model.pth')
        checkpoint = torch.load('/home/murong/code/XAI/UDA-pytorch_mod/temp_model/webvision/beitv2_webvision/finetune1_imagenet_pt/model.pth')
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
        elif model_type == "beitv2_animal10n":    
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
    def __init__(self, root_dir, mode, transform_fn, indices=None, soft_labels=None):
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
            
        if indices is not None:
            self.train_imgs = [self.train_imgs[i] for i in indices]
            self.train_labels = [self.train_labels[i] for i in indices]
            
        if soft_labels is not None:
            self.train_labels = soft_labels

    def __getitem__(self, index):
        if self.mode == 'train':
            img_id, _ = self.train_imgs[index]
            img_path = os.path.join(self.train_dir, img_id)
            target = self.train_labels[index]
        else:  # 'test'
            img_id, _ = self.test_imgs[index]
            img_path = os.path.join(self.test_dir, img_id)
            target = self.test_labels[index]

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
    def __init__(self, root_dir, mode, num_class, transform_fn, indices=None, soft_labels=None):
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
            if indices is not None:
                self.train_imgs = [self.train_imgs[i] for i in indices]
                self.train_labels = [self.train_labels[i] for i in indices]
            if soft_labels is not None:
                self.train_labels = soft_labels

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

# --------------------------
#   4) MLP Classifier
# --------------------------

# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes, num_layers=1, hidden_dim=512, dropout=0.0):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             num_classes (int): Number of output classes.
#             num_layers (int): Number of layers (including output layer).
#             hidden_dim (int): Number of units in each hidden layer.
#             dropout (float): Dropout probability.
#         """
#         super(MLPClassifier, self).__init__()
        
#         layers = []
        
#         # First hidden layer
#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout))
        
#         # Intermediate hidden layers
#         for _ in range(num_layers - 2):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))
            
#         # Output layer
#         layers.append(nn.Linear(hidden_dim, num_classes))
        
#         self.classifier = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.classifier(x)


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
    model_type = "beitv2_webvision"  # Options: "dinov2", "clip", "swav", "esvit"
    # Choose your dataset
    dataset_name = "webvision"  # Options: "cifar10", "animal10N", "webvision"

    # Hyperparameters, etc.
    batch_size = 64
    num_workers = 1
    feature_dir = "./features"

    # -------------------------
    #  Load model + transform
    # -------------------------
    model, transform_fn = load_model_and_transform(model_type)

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
        num_workers = 1
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
    
    
    from tqdm import tqdm

# load selection
selection_flag = True
if selection_flag:
    # selection_method = "disc_selection"
    # selection_method = "inf_selection_ent"
    # clean_ids = np.load(f"/home/mmr/code/XAI/DISC_inf/selection/webvision/{selection_method}/filter_clean_ids.npy")
    # hard_ids = np.load(f"/home/mmr/code/XAI/DISC_inf/selection/webvision/{selection_method}/filter_hard_ids.npy")
    # trainable_ids = np.concatenate((clean_ids, hard_ids))
    # train_features = train_features[trainable_ids]
    # train_labels = train_labels[trainable_ids]
    
    # clean_ids = np.load(f'/home/mmr/code/XAI/UDA-pytorch_mod/softlabel/selection/dinov2/webvision/emb/filter_clean_ids.npy')
    # clean_ids = np.load('/home/mmr/code/XAI/UDA-pytorch_mod/save/webvison/inf_selection_ce/filter_clean_ids.npy')
    
    
    # clean_ids = np.load('/home/murong/code/XAI/UDA-pytorch_mod/softlabel/ood_conft/selection/webvision/selection_after_webvision_clsfinetune_cross/filter_train_ids.npy')
    # clean_labels = np.load('/home/murong/code/XAI/UDA-pytorch_mod/softlabel/ood_conft/selection/webvision/selection_after_webvision_clsfinetune_cross/filter_train_labels.npy')
    
    clean_dir = './softlabel/ood_conft/selection/webvision/selection_after_finetune1_imagenet_pt_knnemb/'
    clean_ids = np.load(f'{clean_dir}/filter_train_ids.npy')
    clean_labels = np.load(f'{clean_dir}/filter_train_labels.npy')
    
    # clean_dir = '/home/murong/code/XAI/UDA-pytorch_mod/softlabel/ood_conft/selection/webvision/webvision_inpretrain_infexpand'
    # clean_ids = np.load(f'{clean_dir}/filter_clean_ids.npy')
    # clean_labels = None
    
    
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomApply(
        #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        #     p=0.8
        # ),
        # transforms.RandomGrayscale(p=0.2),
    ])
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # first global crop
    global_crops_scale = [0.8, 1.0]
    global_transfo1 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
        flip_and_color_jitter,
        normalize,
    ])
    
    train_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    train_dataset = webvision_dataset(root_dir, 'train', 50, train_transform, indices=clean_ids, soft_labels=clean_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)

# class FinetuneBeitV2Model(nn.Module):
#     def __init__(self, model, num_classes):
#         super(FinetuneBeitV2Model, self).__init__()
#         self.model = model
#         self.fc = nn.Linear(model.embed_dim, num_classes)
                    

#     def forward(self, x):
#         bool_masked_pos = torch.zeros((x.shape[0], 196), dtype=torch.bool).to(x.device)
#         x, _ = self.model.forward_features(x, bool_masked_pos)
#         # get cls token
#         # print(x)
#         x = x[:, 0]
#         x = self.fc(x)
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

class Mixup(nn.Module):

    def __init__(self, num_classes=10, alpha=5.0, model=None):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = torch.tensor(alpha).to(self.device)
        if model:
            model.dummy_head = nn.Linear(model.mlp_head[-1].in_features, num_classes, bias=True).to(self.device)
            model.mlp_head.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, data_in, data_out):
        self.features = data_out

    def forward(self, x, y, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = torch.zeros(b, self.num_classes).to(self.device).scatter_(
            1, y.view(-1, 1), 1)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p = model(mixed_x)
        loss = -torch.mean(
            torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss

    def ws_forward(self, wx, sx, y, model):
        b = wx.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = torch.zeros(b, self.num_classes).to(self.device).scatter_(
            1, y.view(-1, 1), 1)
        mixed_x = lam * wx + (1 - lam) * sx[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p = model(mixed_x)
        loss = -torch.mean(
            torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss

    def soft_forward(self, x, p, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        index = torch.randperm(b).to(self.device)
        
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * p + (1 - lam) * p[index]
        
        del x, p
        
        mixed_p = model(mixed_x)
        
        loss = -torch.mean(
            torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y, dim=1))
        return loss

    def dummy_forward(self, x, y, model):
        b = x.size(0)
        lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
        lam = max(lam, 1 - lam)
        index = torch.randperm(b).to(self.device)
        y = torch.zeros(b, self.num_classes).to(self.device).scatter_(
            1, y.view(-1, 1), 1)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_p = model(mixed_x)
        mixed_features = self.features.clone()
        dummy_logits = model.dummy_head(mixed_features)
        loss = -torch.mean(
            torch.sum(F.log_softmax(dummy_logits, dim=1) * mixed_y, dim=1))
        return loss

class FinetuneBeitV2Model(nn.Module):
    def __init__(self, model, num_classes, num_fixed_layers=0, hidden_dim=512):
        super(FinetuneBeitV2Model, self).__init__()
        self.model = model
        self.num_fixed_layers = num_fixed_layers

        # Add LayerNorm and MLP head
        self.layer_norm = nn.LayerNorm(model.norm.normalized_shape)
        self.mlp_head = nn.Sequential(
            nn.Linear(model.norm.normalized_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        # Freeze early layers
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze the early layers of the BEIT model."""
        for idx, block in enumerate(self.model.blocks):
            if idx < self.num_fixed_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):
        bool_masked_pos = torch.zeros((x.shape[0], 196), dtype=torch.bool).to(x.device)
        x, _ = self.model.forward_features(x, bool_masked_pos)
        # Get CLS token and apply LayerNorm
        x = x[:, 0]
        x = self.layer_norm(x)
        # Pass through MLP head
        x = self.mlp_head(x)
        return x

# Define optimizer and scheduler for fine-tuning
def get_optimizer_and_scheduler(model, learning_rate=1e-4, weight_decay=1e-4, num_training_steps=1000, num_warmup_steps=100):
    """
    Define optimizer and learning rate scheduler for fine-tuning.

    Args:
        model (nn.Module): The fine-tuning model.
        learning_rate (float): Base learning rate.
        weight_decay (float): Weight decay for optimizer.
        num_training_steps (int): Total number of training steps.
        num_warmup_steps (int): Number of warmup steps.

    Returns:
        optimizer: AdamW optimizer.
        scheduler: CosineAnnealingLR scheduler with warmup.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import get_scheduler

    # Define optimizer
    optimizer = AdamW([
        {"params": model.model.parameters(), "lr": learning_rate * 0.6},  # Smaller LR for base BEIT model
        {"params": model.mlp_head.parameters(), "lr": learning_rate},    # Larger LR for MLP head
        {"params": model.layer_norm.parameters(), "lr": learning_rate},  # Larger LR for LayerNorm
    ], weight_decay=weight_decay)

    # Define scheduler with warmup
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler

# Fine-tuning function
def fine_tune_model(model, train_dataloader, test_dataloader, num_epochs, device, use_mixup=False, mixup_alpha=5.0):
    """
    Fine-tune the model on a given dataset.

    Args:
        model (nn.Module): The fine-tuning model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        test_dataloader (DataLoader): DataLoader for the testing dataset.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to run the training on (CPU/GPU).
        use_mixup (bool): Whether to use mixup loss during training.
        mixup_alpha (float): Alpha value for mixup if used.
    """
    import torch.nn.functional as F
    # from mixup import Mixup  # Assuming the Mixup class is in a file named mixup.py

    # Define optimizer and scheduler
    total_steps = len(train_dataloader) * num_epochs
    optimizer, scheduler = get_optimizer_and_scheduler(model, num_training_steps=total_steps)

    # Initialize Mixup if needed
    mixup = Mixup(num_classes=model.mlp_head[-1].out_features, alpha=mixup_alpha, model=model) if use_mixup else None

    # Move model to device
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Training phase
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_dataloader)):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            # print(labels.shape)
            
            # Forward pass with or without mixup
            if use_mixup:
                if labels.dim() == 1:
                    loss = mixup.forward(images, labels, model)
                else:
                    loss += mixup.soft_forward(images, labels, model)

            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss / len(train_dataloader):.4f}")

        # Testing phase
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Testing Loss: {test_loss / len(test_dataloader):.4f}, Accuracy: {accuracy:.2f}%")


        
finetune_model = FinetuneBeitV2Model(model, num_classes)
finetune_model = finetune_model.to(device)

# optimizer
# optimizer = torch.optim.Adam(finetune_model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0)


criterion = nn.CrossEntropyLoss()


# fine-tune the model
num_epochs = 15
fine_tune_model(finetune_model, train_loader, test_loader, num_epochs, device, use_mixup=True)

# save model
task_name = "imagenet_pt"
save_dir = os.path.join('./temp_model', dataset_name, model_type, f"finetune1_{task_name}")
os.makedirs(save_dir, exist_ok=True)
torch.save(finetune_model.model.state_dict(), os.path.join(save_dir, "model.pth"))
print(save_dir)

# evaluate top1 and top5 accuracy
finetune_model.eval()
correct = 0
correct_top5 = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = finetune_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # top-5 accuracy
        _, top5 = torch.topk(outputs, 5, dim=1)
        correct_top5 += torch.sum(top5 == labels.view(-1, 1)).item()
        
        
print(f"Top-1 Accuracy: {100 * correct / total:.2f}%", f"Top-5 Accuracy: {100 * correct_top5 / total:.2f}%")

