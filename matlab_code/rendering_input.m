function [rendered_im,foreground_mask]=rendering_input(img,matfile,work_path)

depth_offset=200;
init_pred=matfile;
im=img;


color_info=load([work_path '/util/smpl_vertex_color.mat']);
color_info=color_info.colors;

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
    


    cc1=color_info(meshh,1);
    cc1=reshape(cc1,3,[]);
    cc2=color_info(meshh,2);
    cc2=reshape(cc2,3,[]);
    cc3=color_info(meshh,3);
    cc3=reshape(cc3,3,[]);
%     ccolor=cat(3,cc3/10,cc2,cc1);
     ccolor=cat(3,cc1,cc2,cc3)./255;
    
    

    h=figure(1);
    imshow(black_im);hold on;
    patch_handle=patch(xx,yy ,zz+depth_offset,ccolor, 'EdgeColor', 'none','FaceLighting', 'none');
    
    sim=getframe();
    rendered_im=sim.cdata;
    close all;
    
    foreground_mask=~(rendered_im(:,:,1)==0& rendered_im(:,:,2)==0& rendered_im(:,:,3)==0);
    
end


% % % 
% % % 
% % % 
% % % for mm=1:1:size(mlist,1)
% % %     
% % %     fprintf('%d / %d\n', mm,size(mlist,1));
% % %     res=load([ani_fold mlist(mm).name]);
% % %     
% % %     
% % %        
% % %     weak_trn=init_pred.pred_cam;
% % %     
% % %     depth=2*1000/(224*weak_trn(1));
% % %     depth_ani=depth+res.trans(2);
% % %     scale=2*1000/(224*depth_ani);
% % %         
% % %     vertices=res.vertex;
% % %     
% % % %     figure;
% % % %     scatter3(vertices
% % %     
% % % %     rotmat=ea2dcm([0; 0;1.57]);
% % % %     trans_rot=rotmat*(res.trans)';
% % % %     vertices=vertices*rotmat;
% % %     
% % % %     vis_res=vertices.*weak_trn(1).*(1+res.trans(2)/10);
% % %     vis_res=vertices.*scale;
% % % 
% % %     
% % %     vis_res(:,1:2)=vis_res(:,1:2)+weak_trn(2:3);
% % % 
% % %     %add animation translation
% % %     vis_res(:,1)=vis_res(:,1)+res.trans(1);
% % %     vis_res(:,2)=vis_res(:,2)+res.trans(3);
% % %     
% % %     vis_res(:,1:2)=(vis_res(:,1:2)+1);
% % %     vis_res=vis_res*size(im,1);
% % %     vis_res(:,3)=(vis_res(:,3)*-1);
% % %     vis_res=vis_res.*0.5;
% % %     %
% % %     % figure;
% % %     % imshow(im);hold on;
% % %     % scatter3(vis_res(:,1),vis_res(:,2),vis_res(:,3)+depth_offset);
% % %     
% % %     
% % %     %%%%%%%%%%%%%%%%%%%%%% visualization from image
% % %     im_ori=imread(['./test_images/' imname '.jpg']);
% % %     black_im=zeros(size(im_ori));
% % %     vis_res_ori=vis_res;
% % %     vis_res_ori=vis_res_ori.*crop_info(3);
% % %     vis_res_ori(:,1)=vis_res_ori(:,1)+crop_info(2);
% % %     vis_res_ori(:,2)=vis_res_ori(:,2)+crop_info(1);
% % %     
% % %     %
% % %     
% % %     vertex=vis_res_ori;
% % %     meshh=load('smpl_mesh.txt')+1;
% % %     meshh=meshh';
% % %     
% % %     xx=vertex(meshh,1);
% % %     xx=reshape(xx,3,[]);
% % %     yy=vertex(meshh,2);
% % %     yy=reshape(yy,3,[]);
% % %     zz=vertex(meshh,3);
% % %     zz=reshape(zz,3,[]);
% % %     
% % % 
% % % 
% % %     cc1=color_info(meshh,1);
% % %     cc1=reshape(cc1,3,[]);
% % %     cc2=color_info(meshh,2);
% % %     cc2=reshape(cc2,3,[]);
% % %     cc3=color_info(meshh,3);
% % %     cc3=reshape(cc3,3,[]);
% % % %     ccolor=cat(3,cc3/10,cc2,cc1);
% % %     ccolor=cat(3,cc1,cc2,cc3/10);
% % %     
% % % %     cc=[0.7,0.7,0.7];
% % %     
% % %     close all;
% % %     h=figure(1);
% % %     imshow(black_im);hold on;
% % %     patch_handle=patch(xx,yy ,zz+depth_offset,ccolor, 'EdgeColor', 'none','FaceLighting', 'none');
% % % %     set(patch_handle,'AmbientStrength',0.8,'DiffuseStrength',0.5,'SpecularStrength',0.5);%,'FaceAlpha',0.5);
% % % %     light('Position',[1 1 1]);
% % % 
% % %     data=getframe();
% % %     cdata=data.cdata;
% % % % %     
% % % %     figure;
% % % %     imshow(cdata);
% % % %     
% % %     
% % % % %     figure;
% % % % %     imshow(cdata_swap);
% % %     
% % %     sim=getframe();
% % %     sim=sim.cdata;
% % %     imwrite(sim,[out_fold '/' sprintf('%08d.png',count)]);
% % %     close all;
% % %     count=count+1;
% % %     
% % %     
% % %     
% % %     
% % % end
% % % 
% % % 
% % % 
% % % 
% % % % 
% % % % 
% % % %         cc1=color_info(meshh,1);
% % % %         cc1=reshape(cc1,3,[]);
% % % %         cc2=color_info(meshh,2);
% % % %         cc2=reshape(cc2,3,[]);
% % % %         cc3=color_info(meshh,3);
% % % %         cc3=reshape(cc3,3,[]);
% % % %         ccolor=cat(3,cc1,cc2,cc3);
% % % 
% % % % scatter3(vis_res_ori(:,1),vis_res_ori(:,2),vis_res_ori(:,3)+depth_offset);
% % % % patch(xx,yy,zz+100,cc, 'EdgeColor', 'none','FaceLighting', 'phong');
% % % 
% % % % % fileID=fopen('./chekc.obj','w');
% % % % % for ii=1:size(vis_res_ori,1)
% % % % %     fprintf(fileID,'v %f %f %f\n', vis_res_ori(ii,1),vis_res_ori(ii,2),vis_res_ori(ii,3));
% % % % % end
% % % % %
% % % % % for ii=1:size(mesh_num,1)
% % % % %     fprintf(fileID,'f %d %d %d\n', mesh_num(ii,1)+1,mesh_num(ii,2)+1,mesh_num(ii,3)+1);
% % % % % end
% % % % % fclose(fileID);
% % % % %
% % % % %
