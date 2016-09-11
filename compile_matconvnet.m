%% COMPILE_MATCONVNET
%
% Compile MatConvNet
%
% Hyeonseob Nam, 2015 
%

run /home/lrr/workspace/MyMDNet-master/matconvnet/matlab/vl_setupnn ;
cd /home/lrr/workspace/MyMDNet-master/matconvnet;
vl_compilenn('enableGpu', true, ...
               'CUDAROOT', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc');
cd ..;
