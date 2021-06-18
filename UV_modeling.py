import os

import numpy as np
import cv2 as cv

import torch
from torch.autograd import Variable
import torch.nn as nn


import model_Silnet as Silnet_model
import model_Garnet as Garnet_model
import model_Rendernet as Rendernet_model



def uvTransformDP(input_image, iuv_img, tex_res, fillconst = 0):
    dptex = np.ones((24, tex_res, tex_res, 3))* fillconst
    mdptex = np.ones((24, tex_res, tex_res, 3))* fillconst

    iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]

    mask = np.ones((iuv_img.shape[0], iuv_img.shape[1], 3))


    data = input_image[iuv_img[:, :, 0] > 0]
    mdata = mask[iuv_img[:, :, 0] > 0]


    i = iuv_raw[:, 0] - 1


    u = iuv_raw[:, 1]
    v = iuv_raw[:, 2]

    if max(u)>1:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.


    dptex[i.astype(np.int), np.round(u * (tex_res - 1)).astype(np.int), np.round(v * (tex_res - 1)).astype(np.int)] = data
    mdptex[i.astype(np.int), np.round(u * (tex_res - 1)).astype(np.int), np.round(v * (tex_res - 1)).astype(np.int)] = mdata

    return getCombinedDP(dptex),getCombinedDP(mdptex)


def renderDP(dptex, iuv_image):
    rendered = np.zeros((iuv_image.shape[0], iuv_image.shape[1], dptex.shape[-1]))

    iuv_raw = iuv_image[iuv_image[:, :, 0] > 0]

    i = iuv_raw[:, 0] - 1

    if iuv_raw.dtype == np.uint8 or iuv_raw.max() > 1:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]


    rendered[iuv_image[:, :, 0] > 0] = dptex[
        i.astype(np.int), np.round(u * (dptex.shape[1] - 1)).astype(np.int), np.round(v * (dptex.shape[2] - 1)).astype(
            np.int)]

    return rendered



def getCombinedDP(dptex):
    psize = dptex.shape[1]

    r, c = 4, 6
    combinedtex = np.zeros((psize * r, psize * c, dptex.shape[-1]))

    count = 0
    for i in range(r):
        for j in range(c):
            combinedtex[i * psize:i * psize + psize, j * psize:j * psize + psize] = dptex[count]
            count += 1

    return combinedtex






refer_path= "F:/MPI_project/test_set"


save_uvt="{}/UV_model/UV_map_texture/".format(refer_path)
if os.path.isdir(save_uvt) == False:
   os.mkdir(save_uvt)
save_uvl="{}/UV_model/UV_map_label/".format(refer_path)
if os.path.isdir(save_uvl) == False:
   os.mkdir(save_uvl)
save_uvm="{}/UV_model/UV_map_mask/".format(refer_path)
if os.path.isdir(save_uvm) == False:
   os.mkdir(save_uvm)


save_seg="{}/UV_model/segmentation/".format(refer_path)
if os.path.isdir(save_seg) == False:
   os.mkdir(save_seg)
save_img="{}/UV_model/image/".format(refer_path)
if os.path.isdir(save_img) == False:
   os.mkdir(save_img)
save_mask="{}/UV_model/mask/".format(refer_path)
if os.path.isdir(save_mask) == False:
   os.mkdir(save_mask)

infer_list="{}/util/UV_modeling/view_list.txt".format(refer_path)
source="source"



class Dataset(torch.utils.data.Dataset):
    def __init__(self):

        test_list= np.genfromtxt(infer_list, dtype=np.str)
        self.data_list= test_list


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):


        flag_ind=int(self.data_list[idx,0])
        pre_name= self.data_list[idx, 1]
        curr_name= self.data_list[idx, 2]
        #

        if flag_ind==1:

            if os.path.exists("{}/data/{}_img_masked.png".format(refer_path, source)):
                img_r = cv.imread("{}/data/{}_img_masked.png".format(refer_path,source))
            else:
                img_r = cv.imread("{}/data/{}_img_masked.jpg".format(refer_path,source))
            body_r = cv.imread("{}/data/{}_bodypose.png".format(refer_path,source))
            cmask_r= cv.imread("{}/data/{}_segment_converted.png".format(refer_path,source))
            mask_r = cv.imread("{}/data/{}_mask.png".format(refer_path,source))/255
            densepose_r= cv.imread("{}/data/{}_densepose.png".format(refer_path,source))
            densepose_r=np.multiply(densepose_r,mask_r)



            uv_tex= np.zeros((512,768,3))
            uv_lab= np.zeros((512,768,3))
            uv_mask= np.zeros((512,768,3))


            body_t= cv.imread("{}/util/UV_modeling/bodypose_map/{}.png".format(refer_path,curr_name))
            densepose_t= cv.imread("{}/util/UV_modeling/densepose/{}.png".format(refer_path,curr_name))


        else:

            img_r = cv.imread("{}/{}.png".format(save_img,pre_name))
            body_r = cv.imread("util/UV_modeling/bodypose_map/{}.png".format(pre_name))
            cmask_r= cv.imread("{}/{}.png".format(save_seg, pre_name))
            mask_r = cv.imread("{}/{}.png".format(save_mask, pre_name))/255
            densepose_r= cv.imread("util/UV_modeling/densepose/{}.png".format(pre_name))



            uv_tex= cv.imread("{}/{}.png".format(save_uvt, pre_name))
            uv_lab= cv.imread("{}/{}.png".format(save_uvl, pre_name))
            uv_mask= cv.imread("{}/{}.png".format(save_uvm, pre_name))/255


            body_t= cv.imread("{}/util/UV_modeling/bodypose_map/{}.png".format(refer_path,curr_name))
            densepose_t= cv.imread("{}/util/UV_modeling/densepose/{}.png".format(refer_path,curr_name))


        dptex1, dpmask1 = uvTransformDP(img_r, densepose_r, 16)
        dptex2, dpmask2 = uvTransformDP(img_r, densepose_r, 32)
        dptex3, dpmask3 = uvTransformDP(img_r, densepose_r, 64)
        dptex4, dpmask4 = uvTransformDP(img_r, densepose_r, 128)


        dim = (768,512 )

        dptex1 = cv.resize(dptex1, dim, interpolation=cv.INTER_NEAREST )
        dptex2 = cv.resize(dptex2, dim, interpolation=cv.INTER_NEAREST )
        dptex3 = cv.resize(dptex3, dim, interpolation=cv.INTER_NEAREST )

        dpseg1, dpmask1 = uvTransformDP(cmask_r, densepose_r, 16)
        dpseg2, dpmask2 = uvTransformDP(cmask_r, densepose_r, 32)
        dpseg3, dpmask3 = uvTransformDP(cmask_r, densepose_r, 64)
        dpseg4, dpmask4 = uvTransformDP(cmask_r, densepose_r, 128)

        dpseg1 = cv.resize(dpseg1, dim, interpolation=cv.INTER_NEAREST )
        dpseg2 = cv.resize(dpseg2, dim, interpolation=cv.INTER_NEAREST )
        dpseg3 = cv.resize(dpseg3, dim, interpolation=cv.INTER_NEAREST )

        dpmask1 = cv.resize(dpmask1, dim, interpolation=cv.INTER_NEAREST)
        dpmask2 = cv.resize(dpmask2, dim, interpolation=cv.INTER_NEAREST)
        dpmask3 = cv.resize(dpmask3, dim, interpolation=cv.INTER_NEAREST)




        # image warping from the source to uv
        warped_tex=np.multiply(np.multiply(np.multiply(dptex1,(1-dpmask2))+dptex2,(1-dpmask3))+dptex3,(1-dpmask4))+dptex4

        warped_seg=np.multiply(np.multiply(np.multiply(dpseg1,(1-dpmask2))+dpseg2,(1-dpmask3))+dpseg3,(1-dpmask4))+dpseg4
        warped_seg= np.round(warped_seg/40)*40


        warped_mask=dpmask1.astype(np.int) | dpmask2.astype(np.int)| dpmask3.astype(np.int)| dpmask4.astype(np.int)
        warped_mask=warped_mask.astype(np.float32)

        kernel = np.ones((2, 2), np.uint8)
        warped_mask= cv.erode(warped_mask, kernel, iterations=1)
        warped_mask= cv.medianBlur(warped_mask, 3) # 512x768x3

        warped_tex=np.multiply(warped_tex, warped_mask)
        warped_seg=np.multiply(warped_seg,warped_mask)



        # combine previous uv
        update_mask=warped_mask-np.multiply(warped_mask, uv_mask)

        combine_tex=uv_tex+ np.multiply(warped_tex, update_mask)
        combine_seg= uv_lab + np.multiply(warped_seg, update_mask)
        combine_mask= uv_mask + update_mask


        cv.imwrite("{}/{}.png".format(save_uvt, curr_name),combine_tex.astype(np.uint8))
        cv.imwrite("{}/{}.png".format(save_uvl, curr_name),combine_seg.astype(np.uint8))
        cv.imwrite("{}/{}.png".format(save_uvm, curr_name),(combine_mask*255).astype(np.uint8))


        if flag_ind==2:
            samples= cv.imread("{}/util/sampler.png".format(refer_path))/255
            hair_samples = cv.imread("{}/util/hair_sampler.png".format(refer_path)) / 255

            img_r_lr = cv.flip(img_r, 1)
            cmask_r_lr = cv.flip(cmask_r, 1)
            body_r_lr = cv.flip(body_r, 1)


            back_init_im=img_r_lr
            back_init_seg=cmask_r_lr
            fmask = (~(back_init_seg[:,:,0:1]== 240)).astype(np.float32)
            fmask3= np.concatenate((fmask,fmask,fmask),axis=2)

            back_init_seg = np.multiply(back_init_seg, fmask3)
            back_init_im = np.multiply(back_init_im, fmask3)

            hair_mask = (back_init_seg == 40).astype(np.float32)

            partial_im_back = np.multiply(back_init_im , samples)
            partial_seg_back= np.multiply(np.multiply(back_init_seg, samples),(~(back_init_seg == 40)).astype(np.float32)) + np.multiply(back_init_seg, (back_init_seg == 40).astype(np.float32))
            partial_seg_back = np.multiply(partial_seg_back, ((1-hair_samples).astype(np.float32))) + hair_samples* 40

            body_mask= np.multiply(np.multiply((body_t[:,:,0:1]==0), body_t[:,:,1:2]==0) , body_t[:,:,2:3]==0)
            body_mask= (~(body_mask)).astype(np.float32)
            body_mask3= np.concatenate((body_mask,body_mask,body_mask), axis=2)

            partial_im_back = np.multiply(partial_im_back,body_mask3 )
            partial_seg_back = np.multiply(partial_seg_back, body_mask3)

            hair_im = np.multiply(partial_im_back,  hair_mask)
            hair_im = cv.medianBlur(hair_im.astype(np.uint8), 3)

            pseudo_img = np.multiply(partial_im_back ,((1-hair_mask))) + hair_im
            pseudo_label = np.multiply(partial_seg_back, ((1-hair_mask))) + hair_mask* 40

            pseudo_mask = (pseudo_img[:, :, 0:1] == 0) & (pseudo_img[:, :, 1:2] == 0) & (pseudo_img[:, :, 1:2] == 0)
            pseudo_mask = (~pseudo_mask).astype(np.float32)

        else:
            save_texture = np.zeros((24, 128, 128, 3))
            save_texture_seg = np.zeros((24, 128, 128, 3))
            count = 0

            for ii in range(4):
                for jj in range(6):
                    save_texture[count, :, :, :] = combine_tex[128 * ii: 128 * ii + 128, 128 * jj:128 * jj + 128, :]
                    save_texture_seg[count, :, :, :] = combine_seg[128 * ii: 128 * ii + 128, 128 * jj:128 * jj + 128, :]
                    count += 1

            ### render the pseudo map
            pseudo_img = renderDP(save_texture, densepose_t)
            pseudo_mask = (pseudo_img[:, :, 0:1] == 0) & (pseudo_img[:, :, 1:2] == 0) & (pseudo_img[:, :, 1:2] == 0)
            pseudo_mask = (~pseudo_mask).astype(np.float32)
            pseudo_label = renderDP(save_texture_seg, densepose_t)




        #target
        img_r=img_r.transpose((2, 0, 1)).astype(np.float32) / 255
        body_r=body_r.transpose((2, 0, 1)).astype(np.float32) / 255
        cmask_r=cmask_r.transpose((2, 0, 1)).astype(np.float32) / 255
        cmask_r=cmask_r[0:1,:,:]
        mask_r=mask_r.transpose((2, 0, 1)).astype(np.float32)
        mask_r=mask_r[0:1,:,:]

        body_t=body_t.transpose((2, 0, 1)).astype(np.float32) / 255
        pseudo_img=pseudo_img.transpose((2, 0, 1)).astype(np.float32) / 255
        pseudo_mask=pseudo_mask.transpose((2, 0, 1)).astype(np.float32)
        pseudo_mask=pseudo_mask[0:1,:,:]

        pseudo_label=pseudo_label.transpose((2, 0, 1)).astype(np.float32)/255
        pseudo_label=pseudo_label[0:1,:,:]



        return (img_r,body_r,cmask_r,mask_r,body_t,pseudo_img,pseudo_mask,pseudo_label,curr_name)


def main():
    batchsize_eval=1


    th=nn.Tanh()
    device_ids = [0]  # , 1, 2]


    SE_refer= torch.nn.DataParallel(Silnet_model.Image_Encoder( inchannel=5, num_filters=64), device_ids=device_ids).cuda()
    SE_target= torch.nn.DataParallel(Silnet_model.Image_Encoder( inchannel=3, num_filters=64), device_ids=device_ids).cuda()
    SDecoder= torch.nn.DataParallel(Silnet_model.Mask_decoder(nfilt=64,out_channel=1), device_ids=device_ids).cuda()

    SE_refer.load_state_dict(torch.load("{}/checkpoint/Silnet/E_refer.pkl".format(refer_path)))
    SE_target.load_state_dict(torch.load("{}/checkpoint/Silnet/E_target.pkl".format(refer_path)))
    SDecoder.load_state_dict(torch.load("{}/checkpoint/Silnet/Decoder.pkl".format(refer_path)))



    GEncoder= torch.nn.DataParallel(Garnet_model.Cloth_Encoder( inchannel=8, num_filters=64), device_ids=device_ids).cuda()
    GDecoder= torch.nn.DataParallel(Garnet_model.Cloth_decoder(nfilt=64,out_channel=6), device_ids=device_ids).cuda()

    GEncoder.load_state_dict(torch.load("{}/checkpoint/Garnet/GEncoder.pkl".format(refer_path)))
    GDecoder.load_state_dict(torch.load("{}/checkpoint/Garnet/GDecoder.pkl".format(refer_path)))


    RE_refer= torch.nn.DataParallel(Rendernet_model.Image_Encoder( inchannel=5, num_filters=64), device_ids=device_ids).cuda()
    RE_label= torch.nn.DataParallel(Rendernet_model.Label_Encoder( inchannel=2, num_filters=64), device_ids=device_ids).cuda()
    RE_image= torch.nn.DataParallel(Rendernet_model.Label_Encoder( inchannel=4, num_filters=64), device_ids=device_ids).cuda()
    RDecoder= torch.nn.DataParallel(Rendernet_model.Image_decoder(nfilt=64,out_channel=3), device_ids=device_ids).cuda()

    RE_refer.load_state_dict(torch.load("{}/checkpoint/Rendernet/E_refer.pkl".format(refer_path)))
    RE_label.load_state_dict(torch.load("{}/checkpoint/Rendernet/E_label.pkl".format(refer_path)))
    RE_image.load_state_dict(torch.load("{}/checkpoint/Rendernet/E_image.pkl".format(refer_path)))
    RDecoder.load_state_dict(torch.load("{}/checkpoint/Rendernet/Decoder.pkl".format(refer_path)))

    SE_refer.eval()
    SE_target.eval()
    SDecoder.eval()

    GEncoder.eval()
    GDecoder.eval()

    RE_refer.eval()
    RE_label.eval()
    RE_image.eval()
    RDecoder.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    with torch.no_grad():
        val_iteration=0


        # dataset_val = Dataset()
        # dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batchsize_eval, shuffle=False,
        #                                             num_workers=2)

        dataset_val = Dataset()
        # dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batchsize_eval, shuffle=False,
        #                                             num_workers=1)

        for kk in range(7):
            (img_r,body_r,cmask_r,mask_r,body_t,pseudo_img,pseudo_mask,pseudo_label,curr_name) = dataset_val[kk]

        #
        # for  (img_r,body_r,cmask_r,mask_r,body_t,pseudo_img,pseudo_mask,pseudo_label,curr_name) in dataloader_val:


            img_r = torch.from_numpy(img_r).float().to(device)
            body_r = torch.from_numpy(body_r).float().to(device)
            cmask_r = torch.from_numpy(cmask_r).float().to(device)
            mask_r = torch.from_numpy(mask_r).float().to(device)

            body_t = torch.from_numpy(body_t).float().to(device)

            pseudo_img = torch.from_numpy(pseudo_img).float().to(device)
            pseudo_mask = torch.from_numpy(pseudo_mask).float().to(device)
            pseudo_label = torch.from_numpy(pseudo_label).float().to(device)


            qimg_r= Variable(img_r.cuda()).view(batchsize_eval, -1, 256, 256)
            qbody_r= Variable(body_r.cuda()).view(batchsize_eval, -1, 256, 256)
            qcmask_r= Variable(cmask_r.cuda()).view(batchsize_eval, -1, 256, 256)
            qmask_r = Variable(mask_r.cuda()).view(batchsize_eval, -1, 256, 256)
            qbody_t= Variable(body_t.cuda()).view(batchsize_eval, -1, 256, 256)

            qpseudo_img = Variable(pseudo_img.cuda()).view(batchsize_eval, -1, 256, 256)
            qpseudo_mask= Variable(pseudo_mask.cuda()).view(batchsize_eval, -1, 256, 256)
            qpseudo_label= Variable(pseudo_label.cuda()).view(batchsize_eval, -1, 256, 256)

            bias0= Variable(torch.Tensor(np.ones(([batchsize_eval,1,256,256]))*0.5).cuda(), requires_grad=False)


            ##need to consider the value range

            qstack_input = torch.cat([qbody_r, qmask_r, qcmask_r], 1)
            qbody_guide = torch.cat([qbody_t], 1)

            r5 = SE_refer((qstack_input), 1)
            x1, x2, x3, x4, x5 = SE_target((qbody_guide), 2)

            out_mask= th(SDecoder(r5, x1, x2, x3, x4, x5))
            out_mask=(out_mask>0.5).type(torch.cuda.FloatTensor)


            #predictd clothing
            #need to reconsider the value range.
            qstack_input=torch.cat([qbody_r,qbody_r,qmask_r,qcmask_r],1)
            qbody_guide=torch.cat([qbody_t,qbody_t,out_mask,qpseudo_label],1)
            qstack_input=(qstack_input-bias0)*2
            qbody_guide=(qbody_guide-bias0)*2

            z5 = GEncoder((qstack_input), 1)
            x1, x2, x3, x4, x5 = GEncoder((qbody_guide), 2)
            out_cloth= (GDecoder(z5, x1, x2, x3, x4, x5))
            unmask=out_cloth.detach()
            out_cloth=(out_cloth>0.7).type(torch.cuda.FloatTensor)
            unmask=(unmask<0.7).type(torch.cuda.FloatTensor)

            stack_cloth=out_cloth[:,0:1,:,:]*40
            stack_cloth=stack_cloth*unmask[:,1:2,:,:]+out_cloth[:,1:2,:,:]*80
            stack_cloth=stack_cloth*unmask[:,2:3,:,:]+out_cloth[:,2:3,:,:]*120
            stack_cloth=stack_cloth*unmask[:,3:4,:,:]+out_cloth[:,3:4,:,:]*160
            stack_cloth=stack_cloth*unmask[:,4:5,:,:]+out_cloth[:,4:5,:,:]*200
            stack_cloth=stack_cloth*unmask[:,5:6,:,:]+out_cloth[:,5:6,:,:]*240
            stack_cloth=stack_cloth/255




            qstack_input_256 = torch.cat([qcmask_r, qmask_r, qimg_r], 1)
            label = torch.cat([stack_cloth, out_mask], 1)
            label2 = torch.cat([qpseudo_img, qpseudo_mask], 1)

            qstack_input_256=(qstack_input_256-bias0)*2
            label=(label -bias0)*2
            label2=(label2-bias0)*2


            source_mu, source_logstd = RE_refer((qstack_input_256))

            x1, x2, x3, x4, x5 = RE_label((label), 2)
            f1, f2, f3, f4, f5 = RE_image((label2), 2)

            source_std = source_logstd.exp()
            source_eps = Variable(torch.cuda.FloatTensor(source_std .size()).normal_())
            source_z = source_mu + source_std * source_eps

            out_image = RDecoder(source_z, x1, x2, x3, x4, x5, f1, f2, f3, f4, f5)

            print("now: ", curr_name[:][:], "\n")

            for ii in range(batchsize_eval):
                    sim_mask= out_mask [ii,:,:].view(1,256,256)
                    sim_mask= sim_mask.data.cpu().numpy()
                    # sim_mask=(sim_mask/2+0.5)
                    sim_mask=sim_mask*255
                    sim_mask=np.concatenate((sim_mask, sim_mask,sim_mask), axis=0)
                    sim_mask=sim_mask.transpose((1,2,0))


                    sim_cloth= stack_cloth[ii,:,:].view(1,256,256)
                    sim_cloth= torch.cat([sim_cloth,sim_cloth,sim_cloth],0)
                    sim_cloth= sim_cloth.data.cpu().numpy()
                    sim_cloth=sim_cloth*255
                    sim_cloth=(np.round(sim_cloth/40)*40).astype(np.uint8)
                    sim_cloth=sim_cloth.transpose((1, 2, 0))


                    sim_image= out_image[ii,:,:].view(3,256,256)
                    sim_image= sim_image.data.cpu().numpy()
                    sim_image=(sim_image/2+0.5)*255
                    sim_image=sim_image.transpose((1, 2, 0)).astype(np.uint8)


                    cv.imwrite("{}/{}.png".format(save_mask,curr_name[:][:]), sim_mask)
                    cv.imwrite("{}/{}.png".format(save_seg,curr_name[:][:]), sim_cloth)
                    cv.imwrite("{}/{}.png".format(save_img,curr_name[:][:]), sim_image)







if __name__ == '__main__':
    main()








