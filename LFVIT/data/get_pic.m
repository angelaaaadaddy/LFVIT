path = '/Users/mianmaokuchuanma/database/win5-lid/Win5-LID/Distorted/real/*.bmp';
save_path = '/Users/mianmaokuchuanma/database/win5-lid/Win5-LID/SAIs/';
pics = dir(path);
piclist = {pics.name};
for k = 1:size(pics)
    pic_path = strcat(pics(k).folder, '/', pics(k).name);
    dis_lf = imread(pic_path);
    dis_lf = permute(reshape(dis_lf,[9, 434, 9, 625, 3]),[1,3,2,4,5]);
    for i = 1:9
        for j = 1:9
            save_pic = squeeze(dis_lf(i,j,:,:,:));
            idx = (i - 1) * 9 + j;
            pic_name = strcat(pics(k).name(1:end - 4), '_', num2str(idx), '.bmp');
            imwrite(save_pic, strcat(save_path, pic_name));
        end
    end
end



% dis_img_path = './HEVC_Bikes_44.bmp';
% dis_lf = imread(dis_img_path);
% dis_lf = permute(reshape(dis_lf,[9, 434, 9, 625, 3]),[1,3,2,4,5]);
% 
% save_path = './pic';
% 
% for i = 1:9
%     for j = 1:9
%         save_pic = squeeze(dis_lf(i,j,:,:,:));
%         idx = (i-1) * 9 + j;
%         imwrite(save_pic, strcat(save_path, '/', num2str(idx), '.png'));
%     end
% end
