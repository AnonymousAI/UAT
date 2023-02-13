import argparse
import os

import torch
import torchvision
from torch_fidelity import calculate_metrics
import numpy as np
from ipdb import set_trace
import shutil
from tqdm import tqdm
import time
import model
from tensor_transforms import convert_to_coord_format
from distributed import get_rank
from torchvision import transforms

def get_Text_From_TediGan_val30000(wordtoix,device):
    tediGan_text_path = '../valid_text_30000/'
    all_text_fileName = os.listdir(tediGan_text_path)
    all_text_fileName.sort()
    text_list = []
    cations_list = []
    caption_len = []
    for name in all_text_fileName:
        f = open( tediGan_text_path + name, 'r' )
        file_content = f.read()
        file_content_list = file_content.split(' ')
        text_list.append(file_content_list)
        f.close()
    for t in text_list:
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        if len(rev)<30:
            caption_len.append(len(rev))
            while len(rev)<30:
                rev.append(0)
        else:
            caption_len.append(30)
            rev = rev[:30]
            rev[-1]=0
        rev = torch.tensor(rev)
        rev = rev.to(device)
        cations_list.append(rev)
    return cations_list,all_text_fileName,caption_len

from torchvision import utils
def save_grid_images(images, filename,img_dir):
    n_sample = images.size(0)
    utils.save_image(
        images,
        f"{img_dir}/{filename}",
        nrow=int(n_sample ** 0.5),
        normalize=True,
        range=(-1, 1),
    )

@torch.no_grad()
def calculate_fid_CLIP_with_TediGan_text(model, val_dataset,train_dataset, bs, textEnc, num_batches, latent_size,data_iter,
                    prepare_data,get_text_input,word2id,get_text,
                  val_loader,save_dir='fid_imgs', device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    save_text_path = save_dir + '/text/'
    os.makedirs(save_text_path, exist_ok=True)
    caption_list,all_text_fileName,_ = get_Text_From_TediGan_val30000(word2id,device)
    # torch.tensor(caption_list)
    # caption_list = caption_list.to(device)
    for i in tqdm(range(num_batches)):
        # try:
        #     data = data_iter.next()
        # except:
        #     data_iter = iter(val_loader)
        #     data = data_iter.next()

        # imgs, caps, cap_lens, _, keys = prepare_data(data)
        caps = caption_list[i*bs:(i+1)*bs]
        save_texts = get_text(caps)
        keys = all_text_fileName[i*bs:(i+1)*bs]
        # FOR CLIP
        texts = get_text_input(caps)
        states = textEnc.encode_text(texts).float()
        states = states.detach()

        fake_imgs, _, _, _ = model(states)
        for j in range(bs):
            cnt += 1
            img_name = f"{keys[j]}_{str(cnt).zfill(6)}.png"
            torchvision.utils.save_image(fake_imgs[j, :, :, :],
                                        #  os.path.join(save_dir, '00000.png'), range=(-1, 1),
                                         os.path.join(save_dir, img_name), range=(-1, 1),
                                         normalize=True)
            text_name = f"{keys[j]}_{str(cnt).zfill(6)}.txt"
            file = open(os.path.join(save_text_path, text_name),'w')
            file.write(save_texts[j])
            file.close()
    metrics_dict1 = calculate_metrics(input1=save_dir, input2=train_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    metrics_dict2 = calculate_metrics(input1=save_dir, input2=val_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    if os.path.exists(save_dir) is not None:
        shutil.rmtree(save_dir)
    return metrics_dict1,metrics_dict2
