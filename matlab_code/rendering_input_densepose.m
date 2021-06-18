function [rendered_im]=rendering_input_densepose(img,matfile,work_path)

depth_offset=200;
init_pred=matfile;
im=img;

color_info=load([work_path '/util/smpl_color_info.mat']);


color_info=color_info.color_info;

weak_trn=init_pred.pred_cam;
vis_res=init_pred.pred_vertices;

    vis_res=vis_res.*weak_trn(1);

    vis_res(:,1:2)=vis_res(:,1:2)+weak_trn(2:3);

    %add animation translation
    
    vis_res(:,1:2)=(vis_res(:,1:2)+1);
    vis_res=vis_res*size(im,1);
    vis_res(:,3)=(vis_res(:,3)*-1);
    vis_res=vis_res.*0.5;
    
    
    
    
    black_im=uint8(zeros(size(im)));
  
    
    %
    
    vertex=vis_res;
    meshh=load([work_path '/util/smpl_mesh.txt'])+1;
    meshh=meshh';
    
    xx=vertex(meshh,1);
    xx=reshape(xx,3,[]);
    yy=vertex(meshh,2);
    yy=reshape(yy,3,[]);
    zz=vertex(meshh,3);
    zz=reshape(zz,3,[]);
    

    cc=load([work_path '/util/smpl_densepose_info.mat']);
    id=cc.texture_info(:,1);
    color=cc.densepose;

    cc2=color(:,1:2:5)'.*255;
    cc1=color(:,2:2:6)'.*255;
    cc3=[id' ; id' ; id'];
    ccolor=cat(3,cc1,cc2,cc3.*10);

    h=figure(1);
    imshow(black_im);hold on;
    patch_handle=patch(xx,yy ,zz+depth_offset,ccolor./255, 'EdgeColor', 'none','FaceLighting', 'none');
    
    sim=getframe();
    rendered_im=sim.cdata;
    close all;
    
% %     
% %     black_im_big=uint8(zeros(size(im,1)*3/2,size(im,1)*3/2));
% %     
% %     h=figure(1);
% %     imshow(black_im_big);hold on;
% %     patch_handle=patch(xx*3/2,yy*3/2 ,zz*3/2+depth_offset,ccolor./255, 'EdgeColor', 'none','FaceLighting', 'none');
% %     
% %     sim_big=getframe();
% %     rendered_im_big=sim_big.cdata;
% %     close all;
% %     
% % %     foreground_mask=~(rendered_im(:,:,1)==0& rendered_im(:,:,2)==0& rendered_im(:,:,3)==0);
    
end
