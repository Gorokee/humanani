work_path='./';
cd(work_path);
addpath(genpath([work_path 'matlab_code/']));

img_name='source';




%%
%convert the garment label detection results to our representation.
segment=imread([work_path '/data/' img_name '_segment.png']);
im=imread([work_path '/data/' img_name '_img.png']);
    
    
mask1=segment==2| segment==1;
mask2=segment==5 | segment==6 | segment==7 |  segment==3;
mask3=segment==9 | segment==12 ;
mask4=segment==8 | segment==18 | segment==19;
mask5=segment==4 | segment==14 | segment==15 | segment==16 | segment==10| segment==17 ;
mask6=segment==13 ;

cloth=uint8(mask1).*40+uint8(mask2).*80+uint8(mask3).*120+uint8(mask4).*160+uint8(mask5).*200+uint8(mask6).*240;
cloth3=cat(3,cloth,cloth,cloth);
mask=uint8(segment>0);
mask33=cat(3,mask,mask,mask);
mask33_ero=imerode(mask33,strel('disk',1));    

masked_img=im.*(mask33_ero);

imwrite(cloth.*mask33_ero(:,:,1),[work_path '/data/' img_name '_segment_converted.png']);
imwrite(masked_img,[work_path  '/data/' img_name   '_img_masked.png']);
imwrite(mask33_ero.*255,[work_path  '/data/' img_name '_mask.png']);



%%
%convert the body fitting results to our representation.

img=imread([work_path '/data/' img_name '_img_masked.png']);
matfile=load([work_path  '/data/' img_name  '_bodyfit.mat']);
[rendered,mask]=rendering_input(img,matfile,work_path);
[rendered_dp]=rendering_input_densepose(img,matfile,work_path);


save_dpose=uint8(zeros(256,256,3));
for kk=1:24
    mask=rendered_dp(:,:,3)==kk*10;
    mask=uint8(imdilate(imerode(mask,strel('disk',1)),strel('disk',1)));
    mask=cat(3,mask,mask,mask);
    save_dpose=save_dpose+rendered_dp.*mask;
end

save_dpose(:,:,3)=save_dpose(:,:,3)./10;


invalid_mask=masked_img(:,:,1)==0 & masked_img(:,:,2)==0 & masked_img(:,:,3)==0;
valid_mask=~invalid_mask;
save_dpose=save_dpose.*uint8(valid_mask);


imwrite(rendered,[work_path '/data/' img_name '_bodypose.png']);
imwrite(save_dpose,[work_path '/data/' img_name  '_densepose.png']);



