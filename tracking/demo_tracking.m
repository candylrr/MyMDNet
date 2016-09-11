%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%
%cleanup = onCleanup(@() exit() );

%this is a visual tracking code which improves fps performance of MDNet
%you can select a seqname to test our algorithm from 
%         {'basketball','carScale','bolt','car4','coke','couple',
%        'crossing','david','david2','david3','deer','dog1','dudek','faceocc1',
%        'faceocc2','football1','freeman1','freeman3',
%        'freeman4','girl','ironman','jumping','lemming','liquor','matrix','motorRolling',
%        'mountainBike','singer1','singer2','skiing','subway',
%        'suv','sylvester','tiger1','tiger2','walking','walking2','woman',
%        'boy','carDark','fish','fleetface','football','shaking','skating1','soccer','trllis'}
clear;
seqname='Dog';
fw=fopen('/home/lrr/workspace/MyMDNet-master/tracking/testfile.txt','w');
fprintf(fw,'%s',seqname);
fclose(fw);
system('python /home/lrr/workspace/MyMDNet-master/tracking/allcluster.py');

result=importdata('/home/lrr/workspace/MyMDNet-master/tracking/resultfile.txt');
ff=fopen('/home/lrr/workspace/MyMDNet-master/tracking/resultfile.txt','w');
fclose(ff);
%result=py('/home/lrr/workspace/MyMDNet-masterautomatic/similarity/allcluster.py',seqname);
%[status,result]=py('python /home/lrr/workspace/MyMDNet-masterautomatic/similarity/allcluster.py',seqname);
if result~=0
    fprintf('select branch:%d\n',result);
    getfilter(result);
else
    fprintf('select branch:18\n');
end
%fprintf('select branch:%d\n',n);
conf = genConfig('otb',seqname);
% conf = genConfig('vot2015','ball1');
tt=tic;
%net = fullfile('models','mdnet_2_new.mat');
if result~=0
    net = fullfile('models','mymdnet_otbinuse.mat');
    mdnet_run(conf.imgList, conf.gt(1,:), net);
else
    net=fullfile('models','mdnet_vot-otb.mat');
    mdnet_runnobranch(conf.imgList, conf.gt(1,:), net);
end
%net = fullfile('models','mymdnet_myotb.mat');
tt=toc(tt);
fprintf('%f seconds in all,process\n',tt);