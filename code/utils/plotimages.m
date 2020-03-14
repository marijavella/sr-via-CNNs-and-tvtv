function [] = plotimages(j,im_HR, im_gt, im_w, tvtvimage, sf)
% This function is used to plot the GT, HR from a learning-based method (e.g., CNN) and the resulting image using our method. 
% The first two channels are upscaled using bicubic interpolation while the third layer represents the super-resolved output.

if size(im_HR,3) >1
ycbcrim = rgb2ycbcr(im_HR);
im_CB(j).out = ycbcrim(:,:,2);
im_CB(j).out = imresize(im_CB(j).out,1/sf);
im_CB(j).out = imresize(im_CB(j).out,sf);
im_CR(j).out = ycbcrim(:,:,3);
im_CR(j).out = imresize(im_CR(j).out,1/sf);
im_CR(j).out = imresize(im_CR(j).out,sf);
end

if size(im_HR,3) >1
figure;
subplot(131); imshow(ycbcr2rgb(cat(3,im_gt,im_CB(j).out,im_CR(j).out))); title('Ground Truth');
subplot(132); imshow(ycbcr2rgb(cat(3,im_w,im_CB(j).out,im_CR(j).out))); title('From CNN');
subplot(133); imshow(ycbcr2rgb(cat(3,(tvtvimage),im_CB(j).out,im_CR(j).out))) ; title('TVTV Solver'); 
else
figure;
subplot(131); imshow((im_gt)); title('Ground Truth');
subplot(132); imshow((im_w)); title('From CNN');
subplot(133); imshow((tvtvimage)); title('TVTV Solver');
end

end

