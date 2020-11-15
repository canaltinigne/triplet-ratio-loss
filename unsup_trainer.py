import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np


def train(model, trainLoader, optimizer, loss_fn, t_losses, m, ep, cfg, centers, kmeans, center_distances, c_points, class_ids):
    model.train()
    
    with tqdm(total=len(trainLoader), dynamic_ncols=True) as progress:
        progress.set_description(f'Epoch {ep+1}')

        t_loss = 0
        metric_loss = 0
        
        rotation_loss = 0
        q_loss = 0
        p_loss = 0

        for idx, inputs in enumerate(trainLoader):

            optimizer.zero_grad()
            
            q_images = inputs['query'].cuda()
            target = inputs['target']
            
            #p_images = inputs['positive'].cuda()
            
            #batch_size = q_images.size()[0]
            #random_indices = [np.random.choice([y for y in range(batch_size) if y != x]) for x in range(batch_size)]
            
            q_out, q_recon = model.forward(q_images)
            c_points.append(q_out.cpu().data)
            
            #p_out, p_rot, p_recon = model.forward(p_images)
            #n_out = q_out[random_indices] 
            
            #_, rot_pred = model.forward(r_images.cuda())
            
            m_loss = loss_fn(q_out, target, centers, kmeans, center_distances, m, class_ids[idx])
            #rot_loss = nn.CrossEntropyLoss()(p_rot, inputs['rotation'].cuda())
            q_recon_loss = nn.MSELoss()(q_recon, q_images)
            #p_recon_loss = nn.MSELoss()(p_recon, p_images)
            
            #ce = nn.CrossEntropyLoss()(rot_pred, inputs['rotation'].cuda())
            
            loss = m_loss + q_recon_loss #+ rot_loss +  + p_recon_loss
            loss.backward()

            optimizer.step()

            t_loss += loss.item()
            metric_loss += m_loss.item()
            #rotation_loss += rot_loss.item()
            q_loss += q_recon_loss.item()
            #p_loss += p_recon_loss.item()
            #ce_loss += ce.item()

            avg_loss = t_loss / (idx + 1)
            avg_metric = metric_loss / (idx + 1)
            #avg_rotate = rotation_loss / (idx + 1)
            avg_recon_q = q_loss / (idx + 1)
            #avg_recon_p = p_loss / (idx + 1)
            #avg_ce = ce_loss / (idx + 1)
            
            progress.update(1)
            progress.set_postfix(loss=avg_loss, m_val=m, metric_loss=avg_metric, q_recon = avg_recon_q)
                                 #rotation=avg_rotate, , p_recon = avg_recon_p)

    avg_t_loss = t_loss / len(trainLoader)
    t_losses.append(avg_t_loss)
    
    return avg_t_loss