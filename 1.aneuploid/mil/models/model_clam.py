import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class ConvFlattenBlock(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048, kernel_size=7):
        super(ConvFlattenBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Permute input to match Conv2D format: (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)  # (batch_size, 2048, 7, 7)
        x = self.conv(x)
        x = self.flatten(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout, no_activation = False):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_features, out_features)]
        if not no_activation:
            layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(0.25))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
   

class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, input_L=1024, input_dim=2, first_layer_attention = False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [input_L, 512, 256], "big": [input_L, 512, 384]}
        size = self.size_dict[size_arg]

        fc = []
        if input_dim == 4:
            fc.append(ConvFlattenBlock(in_channels=2048, out_channels=2048, kernel_size=7))
        if size[0] > 1024:
            fc.append(LinearBlock(size[0], 1024, dropout = not first_layer_attention, no_activation = first_layer_attention)) #reconsider to remove the dropout here.
        fc.append(LinearBlock(1024, size[1], dropout))

        # fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        # if dropout:
        #     fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        # Return probabilities instead of class predictions
        # all_logits = classifier(A)
        # probabilities = classifier(all_logits, dim=1)  # Get probabilities
        # class_1_probs = probabilities[:, 1]

        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)

        # Return probabilities instead of class predictions
        all_logits = classifier(A)
        probabilities = F.softmax(all_logits, dim=1)  # Get probabilities
        # class_1_probs = probabilities[:, 1]

        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets, probabilities

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, tile_prob=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK   
        A_raw = A   
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N
        # if probability_only:
        #     probabilities = self.instance_classifiers[1](A)
        #     return probabilities
        
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            # all_c1_probs = []  # To store tile-level probabilities
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    # all_c1_probs.extend(probabilities.cpu().numpy())  # Store probabilities
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets, c1_probs = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                        all_c1_probs.extend(c1_probs.cpu().numpy())  # Store probabilities
                    else:
                        continue
                total_inst_loss += instance_loss
            # if probability_only:
            #     return np.array(all_c1_probs)

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds) }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        if tile_prob:
            # A = A.view(-1, 1)
            A_t = torch.transpose(A, 1, 0)
            # print(A.shape)
            # print(h.shape)
            weighted_h = A_raw * h
            # summed_tensor = torch.sum(weighted_h, dim=0, keepdim=True)

            # inst_logits = self.instance_classifiers[1](summed_tensor)
            h_logits = self.classifiers(h)
            h_probs = F.softmax(h_logits, dim = 1)
            formatted_hprobs = [[f"{value:.4f}" for value in row] for row in h_probs.cpu().numpy()]
            formatted_hprobs = np.array(formatted_hprobs).astype(np.float32)
            formatted_hprobs = formatted_hprobs[:, 1]

            A_h_logits = self.classifiers(weighted_h)
            A_h_probs = F.softmax(A_h_logits, dim = 1)
            formatted_Ahprobs = [[f"{value:.4f}" for value in row] for row in A_h_probs.cpu().numpy()]
            # print(formatted_Ahprobs)
            formatted_Ahprobs = np.array(formatted_Ahprobs).astype(np.float32)
            formatted_Ahprobs = formatted_Ahprobs[:, 1]

            A_raw_np = np.array(A_raw.cpu()).astype(np.float32)
            print("A_raw_np = ", A_raw_np)


            # results_dict = {'h_probs': formatted_hprobs, "A": A_t, "h_weighted" : formatted_Ahprobs, "A_raw": A_raw_np}
            results_dict = {"A_raw": A_raw_np}

        return logits, Y_prob, Y_hat, A, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, input_L=1024, input_dim=2, first_layer_attention = False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [input_L, 512, 256], "big": [input_L, 512, 384]}
        size = self.size_dict[size_arg]
        fc = []
        if input_dim == 4:
            fc.append(ConvFlattenBlock(in_channels=2048, out_channels=2048, kernel_size=7))
        if size[0] > 1024:
            fc.append(LinearBlock(size[0], 1024, dropout = not first_layer_attention, no_activation = first_layer_attention))
        fc.append(LinearBlock(1024, size[1], dropout))

        # fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        # if dropout:
        #     fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, tile_prob=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK   
        A_raw = A     
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            # all_c1_probs = []  # To store tile-level probabilities
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    # all_c1_probs.extend(c1_probs.cpu().numpy())  # Store probabilities
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                        # all_c1_probs.extend(c1_probs.cpu().numpy())  # Store probabilities
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds) }
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        if tile_prob:
            # A = A.view(-1, 1)
            A_t = torch.transpose(A, 1, 0)
            # print(A_raw.shape)
            # print(h.shape)
            # weighted_h= torch.empty(1, self.n_classes).float().to(device)
            # for c in range(self.n_classes):
            #     weighted_h[:, c] = A_raw[:, c] * h
            # # summed_tensor = torch.sum(weighted_h, dim=0, keepdim=True)

            # # inst_logits = self.instance_classifiers[1](summed_tensor)
            # # h_logits = self.classifiers(h)
            # h_logits = torch.empty(1, self.n_classes).float().to(device)
            # for c in range(self.n_classes):
            #     h_logits[0, c] = self.classifiers[c](h[c])
            # h_probs = F.softmax(h_logits, dim = 1)
            # formatted_hprobs = [[f"{value:.4f}" for value in row] for row in h_probs.cpu().numpy()]
            # formatted_hprobs = np.array(formatted_hprobs).astype(np.float32)
            # formatted_hprobs = formatted_hprobs[:, 1]
            # # formatted_hprobs = [[f"{value:.4f}" for value in row] for row in h_probs.numpy()]
            # # formatted_hprobs = np.array(formatted_hprobs).astype(float)

            # # A_h_logits = self.classifiers(weighted_h)
            # A_h_logits = torch.empty(1, self.n_classes).float().to(device)
            # for c in range(self.n_classes):
            #     A_h_logits[0, c] = self.classifiers[c](weighted_h[c])
            # A_h_probs = F.softmax(A_h_logits, dim = 1)
            # formatted_Ahprobs = [[f"{value:.4f}" for value in row] for row in A_h_probs.cpu().numpy()]
            # formatted_Ahprobs = np.array(formatted_Ahprobs).astype(np.float32)
            # formatted_Ahprobs = formatted_Ahprobs[:, 1]
            # # formatted_Ahprobs = [[f"{value:.4f}" for value in row] for row in A_h_probs.numpy()]
            # # formatted_Ahprobs = np.array(formatted_Ahprobs)
            A_raw_np = np.array(A_raw.cpu()).astype(np.float32)

            results_dict = {"A_raw": A_raw_np} #, 'h_probs': formatted_hprobs, "h_weighted" : formatted_Ahprobs, "A_raw": A_raw_np}

        return logits, Y_prob, Y_hat, A, results_dict
