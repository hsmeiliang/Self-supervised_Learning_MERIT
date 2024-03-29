import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):

    def __init__(self, 
                  gnn,
                  projection_hidden_size,
                  projection_size):
        
        super().__init__()
        
        self.gnn = gnn
        self.projector = MLP(512, projection_size, projection_hidden_size)           
        
    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections

    
class EMA():
    
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def sim(x, y):
    z1 = F.normalize(x, dim=-1, p=2)
    z2 = F.normalize(y, dim=-1, p=2)
    return torch.mm(z1, z2.t())


def contrastive_loss_cross_view(pred_1, pred_2, target):
    func = lambda x: torch.exp(x)
    intra_sim = func(sim(pred_1, pred_1))
    inter_sim = func(sim(pred_1, pred_2))
    return -torch.log(inter_sim.diag() / (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))


def contrastive_loss_cross_network(pred_1, pred_2, target):
    func = lambda x: torch.exp(x)
    cross_sim = func(sim(pred_1, target))
    return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))


class MERIT(nn.Module):
    
    def __init__(self, 
                 gnn,
                 feat_size,
                 projection_size, 
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta):
        
        super().__init__()

        self.online_encoder = GraphEncoder(gnn, projection_hidden_size, projection_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, prediction_size, prediction_hidden_size)
        self.beta = beta
                   
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2, sparse):
        ## TODO: 
        # Given training instances: aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2, hyperparameters: beta
        # Please implement the main algorithm of merit using online_encoder and target_encoder 
        # hint: use self.online_encoder, self.online_predictor, self.target_encoder
        # hint2: remember to detach target network

        # The output should be "the calculated overall contrastive loss of MERIT"
        # Therefore, you should implement your "contrative loss function" first
        # For the CL term, please refer to the released hw powerpoint or the original paper for details

        online_input_1 = self.online_encoder(aug_adj_1, aug_feat_1, sparse)
        online_input_2 = self.online_encoder(aug_adj_2, aug_feat_2, sparse)
        
        online_pred_1 = self.online_predictor(online_input_1)
        online_pred_2 = self.online_predictor(online_input_2)

        with torch.no_grad():
            target_output_1 = self.target_encoder(aug_adj_1, aug_feat_1, sparse)
            target_output_2 = self.target_encoder(aug_adj_2, aug_feat_2, sparse)
        
        loss_1 = self.beta * contrastive_loss_cross_view(online_pred_1, online_pred_2, target_output_2.detach()) + (1 - self.beta) * contrastive_loss_cross_network(online_input_1, online_input_2, target_output_2.detach())
        loss_2 = self.beta * contrastive_loss_cross_view(online_pred_2, online_pred_1, target_output_1.detach()) + (1 - self.beta) * contrastive_loss_cross_network(online_input_2, online_input_1, target_output_1.detach())

        loss = 0.5 * (loss_1 + loss_2)
            
        return loss.mean()