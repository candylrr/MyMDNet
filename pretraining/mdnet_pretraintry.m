 function [ ] = mdnet_pretraintry( varargin )
% MDNET_PRETRAIN
% Pretrain MDNet from multiple tracking sequences.
%
% Modified from cnn_imagenet() in the MatConvNet library.
% Hyeonseob Nam, 2015
% 

% The list of tracking sequences for training MDNet.
opts.seqsList  = {
    struct('dataset','otb','list','pretraining/seqList/mymdnet_1.txt'),...
    'outFile', fullfile('models','mymdnet_2.mat')};

% The path to the initial network. 
opts.netFile    = fullfile('models','mdnet_init.mat') ;

% The path to the output MDNet model.
%opts.outFile     = fullfile('models','mdnet_vot-otb_new.mat') ;
opts.outFile     = fullfile('models','mymdnet_2.mat') ;
% The directory to store the RoIs for training MDNet.
%opts.roiDir     = fullfile('models','data_vot-otb') ;
opts.roiDir     = fullfile('models','data_2') ;

opts.train.batch_frames     = 8 ; % the number of frames to construct a minibatch.
opts.train.batchSize        = 128 ;
opts.train.batch_pos        = 32;
opts.train.batch_neg        = 96;

opts.train.numCycles        = 100 ; % #cycles (#iterations/#domains)
opts.train.useGpu           = true ;
opts.train.conserveMemory	= true ;
opts.train.learningRate     = 0.0001 ; % x10 for fc4-6

opts.sampling.crop_mode         = 'warp';
opts.sampling.numFetchThreads   = 8 ;
opts.sampling.posRange          = [0.7 1];
opts.sampling.negRange          = [0 0.5];
opts.sampling.input_size        = 107;
opts.sampling.crop_padding      = 16;

opts.sampling.posPerFrame       = 50;
opts.sampling.negPerFrame       = 200;
opts.sampling.scale_factor      = 1.05;
opts.sampling.flip              = false;

opts = vl_argparse(opts, varargin) ;
opts.roiPath  = fullfile(opts.roiDir, 'roidb.mat');
genDir(opts.roiDir) ;

%% Sampling training data
if exist(opts.roiPath,'file')
    load(opts.roiPath) ;
else
    roidb = mdnet_setup_data(opts.seqsList, opts.sampling);
    save(opts.roiPath, 'roidb') ;
end

%% Initializing MDNet
%K = length(roidb);
K=3;
net = mdnet_init_train(opts, K);

%% Training MDNet
fn = @(roidb,img_idx,batch_pos,batch_neg)...
    getBatch(roidb, img_idx, batch_pos, batch_neg, opts.sampling) ;

net = mdnet_train(net, roidb, fn, opts.train) ;

%% Save
%delete % before net=mdnet_finish...
net = mdnet_finish_train(net);
layers = net.layers;

genDir(fileparts(opts.outFile)) ;
save(opts.outFile, 'layers') ;



% -------------------------------------------------------------------------
function [im,labels] = getBatch(roidb, img_idx, batch_pos, batch_neg, opts)
% -------------------------------------------------------------------------
image_paths = {roidb(img_idx).img_path};

pos_boxes = cell2mat({roidb(img_idx).pos_boxes}');
idx = randsample(size(pos_boxes,1),batch_pos);
pos_boxes = pos_boxes(idx,:);
pos_idx = floor((idx-1)/opts.posPerFrame)+1;

neg_boxes = cell2mat({roidb(img_idx).neg_boxes}');
idx = randsample(size(neg_boxes,1),batch_neg);
neg_boxes = neg_boxes(idx,:);
neg_idx = floor((idx-1)/opts.negPerFrame)+1;

boxes = [pos_idx, pos_boxes; neg_idx, neg_boxes];

im = get_batch(image_paths, boxes, opts, ...
    'prefetch', nargout == 0) ;

if(nargout > 0 && opts.flip)
    flip_idx = find(randi([0 1],size(boxes,1),1));
    for i=flip_idx
        im(:,:,:,i) = flip(im(:,:,:,i),2);
    end
end

labels = single([2*ones(numel(pos_idx),1);ones(numel(neg_idx),1)]);



% -------------------------------------------------------------------------
function [ roidb ] = mdnet_setup_data(seqList, opts)
% -------------------------------------------------------------------------

roidb = {};
for D = 1:length(seqList)
    
    dataset = seqList{D}.dataset;
    seqs_train = importdata(seqList{D}.list);
    
    roidb_ = cell(1,length(seqs_train));
    
    for i = 1:length(seqs_train)
        seq = seqs_train{i};
        fprintf('sampling %s:%s ...\n', dataset, seq);
        
        config = genConfig(dataset, seq);
        roidb_{i} = seq2roidb(config, opts);


    end
    roidb = [roidb,roidb_];
end



% -------------------------------------------------------------------------
function [ net ] = mdnet_init_train( opts, K )
% -------------------------------------------------------------------------
net = load(opts.netFile);
net.layers = net.layers(1:end-2);

% domain-specific layers
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc6', ...
    'filters', 0.01 * randn(1,1,512,2*K,'single'), ...
    'biases', zeros(1, 2*K, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 10, ...
    'biasesLearningRate', 20, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 0) ;
%net.layers{end+1} = struct('type', 'softmaxloss_k', 'name', 'loss') ;
% -------------------------------------------------------------------------

function [ net ] = mdnet_finish_train( net )
% -------------------------------------------------------------------------

%net.layers = net.layers(1:end-2);
net.layers = net;
%{
for i=1:numel(net.layers)
    switch (net.layers{i}.type)
        case 'conv'
            net.layers{i}.filtersLearningRate = 1;
            net.layers{i}.biasesLearningRate = 2;
    end
end
%}
% new domain-specific layer
%{
%old code MDNet
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc6', ...
    'filters', 0.01 * randn(1,1,512,2,'single'), ...
    'biases', zeros(1, 2, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 10, ...
    'biasesLearningRate', 20, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
%}
%change to this to create a new last layer,val is defined before it
%{
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc6', ...
    'filters', net.layers{end+1}.filters(), ...
    'biases', zeros(1, 2, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 10, ...
    'biasesLearningRate', 20, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
%}
old=load('filter.mat');
filter=old.layers.layers{1,17}.filters;
save('savefilter.mat',filter);
filter12=old.layers.layers{1,17}.filters{:,:,1:512,1:2};
filter34=old.layers.layers{1,17}.filters{:,:,1:512,3:4};
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc6', ...
    'filters', filter12, ...
    'biases', zeros(1, 2, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 10, ...
    'biasesLearningRate', 20, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
% -------------------------------------------------------------------------
function genDir(path)
% -------------------------------------------------------------------------
if ~exist(path,'dir')
    mkdir(path);
end

