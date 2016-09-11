'''
cluster.py
Uses the Hamming distance between perceptual hashes to surface near-duplicate
images.
To install and run:
1. pip install imagehash
2. Put some .dat files in a folder someplace (script assumes ./data/imgs/*.dat)
3. python cluster.py
You can adjust the HAMMING_THRESH global to be larger if you want near, but not
identical, dupes.
More information:
Hamming distance: https://en.wikipedia.org/wiki/Hamming_distance
dhash: http://blog.iconfinder.com/detecting-duplicate-images-using-python/
'''

import sys,os, base64, glob, cStringIO, imagehash, itertools, time
from PIL import Image

########## GLOBALS ##########

# Where all your ads are
#IMAGE_DIR = './data/imgs/*.dat'
KEYFRAME_DIR='/home/lrr/workspace/MyMDNet-master/similarity/VOT'


trainingGroup=([['gymnastics1','gymnastics2','iceskater1'],['singer','traffic','helicopter'],['ball',
'iceskater2','woman'],['ball2','polarbear','graduate'],['birds2','road','fish2'],['drunk','tunnel',
'iceskater'],['sunshade','woman','face'],['bmx','ball1','cup'],['blanket','surfing','bag'],['handball1',
'fish3','ctopus'],['hand2','bolt2','book'],['motocross2','car','leaves'],['fish4','fernando','nature'],
['jump','sphere','bag'],['birds1','butterfly','dinosaur'],['fish1','glove','godfather'],['gymnastics3',
'gymnastics4','gymnastics'],['hand1','hand','iceskater1'],['juice','leaves','marching'],['motocross1',
'motocross','pedestrian1'],['pedestrian2','rabbit','racing'],['sheep','singer3','soldier'],['surfing',
'torus','david']])


testname=(['basketball','bolt','boy','car4','carDark','carScale','coke','couple',
        'crossing','david','david2','david3','deer','dog1','dudek','faceocc1',
        'faceocc2','fish','fleetface','football','football1','freeman1','freeman3',
        'freeman4','girl','ironman','jumping','lemming','liquor','matrix','motorRolling',
    'mountainBike','shaking','singer1','singer2','skating1','skiing','soccer','subway',
    'suv','sylvester','tiger1','tiger2','trllis','walking','walking2','woman'])

def hamming(s1, s2):
    '''
    Calculate the normalized Hamming distance between two strings.
    '''
    assert len(s1) == len(s2)
    return float(sum(c1 != c2 for c1, c2 in zip(s1, s2))) / float(len(s1))
########## MAIN ###########
def getmaxindex():
    f=open('/home/lrr/workspace/MyMDNet-master/tracking/testfile.txt','r+')
    m=f.read()
    f.close()
    groupnum=[]
    test_image='/home/lrr/workspace/MyMDNet-master/similarity/test/%s.jpg'%m
    test_hash = str(imagehash.dhash(Image.open(test_image)))
    #test_image='/home/lrr/workspace/MyMDNet-master/similarity/test/3.jpg'
    HAMMING_THRESH = 0.65
    #filejpg=[x for x in os.listdir(test_image) if not os.path.isdir(os.path.join(test_image,x))]  
        #Image_dir=[x for x in os.listdir(KEYFRAME_DIR) if os.path.isdir(os.path.join(KEYFRAME_DIR,x))] 
    for i in range(len(trainingGroup)):
        groupnum.append(0)
        for j in range(3):
            #IMAGE_DIR=os.path.join(KEYFRAME_DIR,trainingGroup[i][j],'_keyframes')
            imgname=trainingGroup[i][j]+'_keyframes'
            IMAGE_DIR=os.path.join(KEYFRAME_DIR,imgname)
            img=os.path.join(IMAGE_DIR,'*.jpg')
            for pic in glob.iglob(img):
                img_hash=str(imagehash.dhash(Image.open(pic)))
                if hamming(img_hash, test_hash) <= HAMMING_THRESH:
		#if float(sum(c1 != c2 for c1, c2 in zip(img_hash, test_hash))) / float(len(img_hash))<= HAMMING_THRESH:
                    groupnum[i]+=1
    if max(groupnum)!=0:
        max_index=groupnum.index(max(groupnum))+1
    else: 
        max_index=0
    #return max_index
    f=open('/home/lrr/workspace/MyMDNet-master/tracking/resultfile.txt','w')
    f.write(str(max_index))
    f.close()
    f=open('/home/lrr/workspace/MyMDNet-master/tracking/testfile.txt','w')
    f.close()
if __name__ == '__main__':
    #m="".join(str(x) for x in sys.argv[1:])
    #print sys.argv[1:]
    getmaxindex()
    #print m
    """
    m=str(sys.argv[1])
    groupnum=[]
    test_image='/home/lrr/workspace/MyMDNet-master/similarity/test/%s.jpg'%m
    test_hash = str(imagehash.dhash(Image.open(test_image)))
    #test_image='/home/lrr/workspace/MyMDNet-master/similarity/test/3.jpg'
    HAMMING_THRESH = 0.65
    #filejpg=[x for x in os.listdir(test_image) if not os.path.isdir(os.path.join(test_image,x))]  
        #Image_dir=[x for x in os.listdir(KEYFRAME_DIR) if os.path.isdir(os.path.join(KEYFRAME_DIR,x))] 
    for i in range(len(trainingGroup)):
        groupnum.append(0)
        for j in range(3):
            #IMAGE_DIR=os.path.join(KEYFRAME_DIR,trainingGroup[i][j],'_keyframes')
            imgname=trainingGroup[i][j]+'_keyframes'
            IMAGE_DIR=os.path.join(KEYFRAME_DIR,imgname)
            img=os.path.join(IMAGE_DIR,'*.jpg')
            for pic in glob.iglob(img):
                img_hash=str(imagehash.dhash(Image.open(pic)))
                if hamming(img_hash, test_hash) <= HAMMING_THRESH:
		#if float(sum(c1 != c2 for c1, c2 in zip(img_hash, test_hash))) / float(len(img_hash))<= HAMMING_THRESH:
                    groupnum[i]+=1
    max_index=groupnum.index(max(groupnum))+1
    print max_index 
    """
    #rint 'end\n'
    
              


