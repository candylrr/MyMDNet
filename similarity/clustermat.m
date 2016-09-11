function max_index=clustermat(m)
KEYFRAME_DIR='/home/smw/workspace/MyMDNettest-masterautomatic/similarity/VOT';
%{
training=['gymnastics1','gymnastics2','iceskater1';'singer','traffic','helicopter';...
'ball','iceskater2','woman';'ball2','polarbear','graduate';'birds2','road','fish2';...
'drunk','tunnel','iceskater';'sunshade','woman','face';'bmx','ball1','cup';'blanket','surfing','bag';...
'handball1','fish3','ctopus';'hand2','bolt2','book';'motocross2','car','leaves';'fish4','fernando','nature';...
'jump','sphere','bag';'birds1','butterfly','dinosaur';'fish1','glove','godfather';...
'gymnastics3','gymnastics4','gymnastics';'hand1','hand','iceskater1';'juice','leaves','marching';...
'motocross1','motocross','pedestrian1';'pedestrian2','rabbit','racing';'sheep','singer3','soldier';...
'surfing','torus','david'];
%}
trainingGroup=[{'gymnastics1','gymnastics2','iceskater1'};
    {'singer','traffic','helicopter'};
    {'ball','iceskater2','woman'};
    {'ball2','polarbear','graduate'};
    {'birds2','road','fish2'};
    {'drunk','tunnel','iceskater'};
    {'sunshade','woman','face'};
    {'bmx','ball1','cup'};
    {'blanket','surfing','bag'};
    {'handball1','fish3','ctopus'};
    {'hand2','bolt2','book'};
    {'motocross2','car','leaves'};
    {'fish4','fernando','nature'};
    {'jump','sphere','bag'};
    {'birds1','butterfly','dinosaur'};
    {'fish1','glove','godfather'};
    {'gymnastics3','gymnastics4','gymnastics'};
    {'hand1','hand','iceskater1'};
    {'juice','leaves','marching'};
    {'motocross1','motocross','pedestrian1'};
    {'pedestrian2','rabbit','racing'};
    {'sheep','singer3','soldier'};
    {'surfing','torus','david'}];
%{
test=['basketball','bolt','boy','car4','carDark','carScale','coke','couple',...
        'crossing','david','david2','david3','deer','dog1','dudek','faceocc1',...
        'faceocc2','fish','fleetface','football','football1','freeman1','freeman3',...
        'freeman4','girl','ironman','jumping','lemming','liquor','matrix','motorRolling',...
    'mountainBike','shaking','singer1','singer2','skating1','skiing','soccer','subway',...
    'suv','sylvester','tiger1','tiger2','trllis','walking','walking2','woman'];
py('set','test',test);
py('set','testname',testname);
getnum1=sprintf('m=test.index(testname)');
py('eval',getnum1);
m=py('get','m');
    %}
strm=strcat(m,'.jpg');
test_image=fullfile('/home/smw/workspace/MyMDNettest-masterautomatic/similarity/test/',strm);
%test_hash = str(imagehash.dhash(imopen(test_image)));
%test_hash=ImageDHash(imread(test_image));

py('set','testimg',test_image);
fprintf('test_imag:%s\n',test_image);
st1=sprintf('import imagehash,cStringIO');
st2=sprintf('from PIL import Image');
st3=sprintf('testhash=str(imagehash.dhash(Image.open(testimg)))');
py('eval',st1);
py('eval',st2);
py('eval',st3);
testhash=py('get','testhash');
%groupnum=zeros(23);
for i=1:23
    groupnum(i)=0;
    for j=1:3
       imgname=strcat(char(trainingGroup(i,j)),'_keyframes/');
       %fprintf('%s\n',imgname);
       img_dir=fullfile(KEYFRAME_DIR,imgname);
       IMAGE_DIR=fullfile(KEYFRAME_DIR,imgname,'*.jpg');
       img=dir(IMAGE_DIR);
       for pic =1:length(img)
           filename=strcat(img_dir,img(pic).name);
           %img_hash=ImageDHash(imread(filename));
           py('set','img',filename);
           stmt1=sprintf('import imagehash,cStringIO');
           stmt2=sprintf('from PIL import Image');
           stmt3=sprintf('imghash=str(imagehash.dhash(Image.open(img)))');
           py('eval',stmt1);
           py('eval',stmt2);
           py('eval',stmt3);
           pichash=py('get','imghash');
           value=pdist2(testhash,pichash,'hamming');
           if value<=0.65
               groupnum(i)=groupnum(i)+1;
           end
        end
    end
end
[max_value,max_index]=max(groupnum);
fprintf('select branch:%d\n',max_index);

    
              


