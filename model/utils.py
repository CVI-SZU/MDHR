import torch
import torch.nn.functional as F

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def get_pred_binary(pred, num_bits):
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    return dec2bin(pred, num_bits)

def calculate_adjmask_bp4d(up_pred, mid_pred, down1_pred, down2_pred):
    # Convert predictions to binary representation
    up_pred_bin = get_pred_binary(up_pred, 4)
    mid_pred_bin = get_pred_binary(mid_pred, 1)
    down1_pred_bin = get_pred_binary(down1_pred, 3)
    down2_pred_bin = get_pred_binary(down2_pred, 4)
    
    # Concatenate the binary representations to form the adj_mask
    adj_mask = torch.cat((
        up_pred_bin[:, :1],   
        up_pred_bin[:, 1:2],  
        up_pred_bin[:, 2:3],  
        mid_pred_bin[:, :1],  
        up_pred_bin[:, 3:4],  
        down2_pred_bin[:, :1],
        down1_pred_bin[:, :1],
        down1_pred_bin[:, 1:2],
        down1_pred_bin[:, 2:3],
        down2_pred_bin[:, 1:2],
        down2_pred_bin[:, 2:3],
        down2_pred_bin[:, 3:4]
        ), dim=1)
    adj_mask = adj_mask.unsqueeze(1).float()
    return adj_mask

def calculate_adjmask_disfa(up_pred, mid_pred, down_pred):
    # Convert predictions to binary representation
    up_pred_bin = get_pred_binary(up_pred, 3)
    mid_pred_bin = get_pred_binary(mid_pred, 2)
    down_pred_bin = get_pred_binary(down_pred, 3)
    
    # Concatenate the binary representations to form the adj_mask
    adj_mask = torch.cat((
        up_pred_bin,   
        mid_pred_bin,  
        down_pred_bin
        ), dim=1)
    adj_mask = adj_mask.unsqueeze(1).float()
    return adj_mask