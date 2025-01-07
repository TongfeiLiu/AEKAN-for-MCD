import torch
import torch.nn as nn
import torch.nn.functional as F
class AEKAN_Loss(nn.Module):
    def __init__(self, lambda_reg):
        super(AEKAN_Loss, self).__init__()
        self.lambda_reg = lambda_reg

    def l2_regularization(self, model):

        l2_reg = torch.tensor(0.0,device=torch.device('cuda'))
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.lambda_reg * l2_reg

    def recon_loss_a(self, adj, adj_pred):
        # 计算重构损失
        loss_restrct_a = F.mse_loss(adj_pred, adj)
        return loss_restrct_a

    def recon_loss_b(self, adj, adj_pred):
        loss_restrct_b = F.mse_loss(adj_pred, adj)
        return loss_restrct_b
    def con_loss_x(self,x_a,x_b):
        con_loss_x=F.mse_loss(x_a,x_b)
        return con_loss_x


    def forward(self, node_t1,node_t2,x_a1, x_a2, x_b1, x_b2,rec_a2,rec_b2,encoder, decoder_a, decoder_b):#函数调用
        #Reconstruction loss
        recon_loss_a2 = self.recon_loss_a(node_t1,rec_a2)
        recon_loss_b2 = self.recon_loss_b(node_t2, rec_b2)
        #Level commonality loss
        con_loss_a1 = self.con_loss_x(x_a1, x_b1)
        con_loss_a2= self.con_loss_x(x_a2, x_b2)
        l2reg_loss = self.l2_regularization(encoder)
        l2reg_loss_da = self.l2_regularization(decoder_a)
        l2reg_loss_db = self.l2_regularization(decoder_b)
        return recon_loss_a2+recon_loss_b2+con_loss_a1+l2reg_loss_db+l2reg_loss_da+l2reg_loss+con_loss_a2
