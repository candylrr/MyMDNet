function []= getfilter(n)
copyfile('/home/lrr/workspace/MyMDNet-master/models/mymdnet_votforotb.mat','/home/lrr/workspace/MyMDNet-master/models/mymdnet_otbinuse.mat')
copyfile('/home/lrr/workspace/MyMDNet-master/models/data_votforotb','/home/lrr/workspace/MyMDNet-master/models/mymdnet_otbinuse')
m=load('/home/lrr/workspace/MyMDNet-master/models/mymdnet_otbinuse.mat');
filters=gather(m.layers.layers{1,17}.filters);
filters=filters(:,:,1:512,n:n+1);
filters=gpuArray(filters);
m.layers.layers{1,17}.filters=filters;
m.layers.layers{1,17}.filters=single(gather(m.layers.layers{1,17}.filters));
m.layers.layers{1, 17}.biases=single(gather(m.layers.layers{1, 17}.biases(1,n:n+1)));
m.layers.layers{1,17}.filtersMomentum=1;
m.layers.layers{1, 17}.biasesMomentum=0;
m.layers.layers{1, 18}.type='softmaxloss';
layers=m.layers;
save('/home/lrr/workspace/MyMDNet-master/models/mymdnet_otbinuse.mat','-append','layers');