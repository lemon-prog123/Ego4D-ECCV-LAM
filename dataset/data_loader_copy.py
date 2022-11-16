import sys, os
from common.utils import get_transform_data
import cv2, json, glob, logging
import torch
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict
import json


logger = logging.getLogger(__name__)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def helper():
    return defaultdict(OrderedDict)


def pad_video(video):
    assert len(video) == 7
    pad_idx = np.all(video == 0, axis=(1, 2, 3))
    mid_idx = int(len(pad_idx) / 2)
    #mid_idx = index
    pad_idx[mid_idx] = False
    pad_frames = video[~pad_idx]
    pad_frames = np.pad(pad_frames, ((sum(pad_idx[:mid_idx]), 0), (0, 0), (0, 0), (0, 0)), mode='edge')
    pad_frames = np.pad(pad_frames, ((0, sum(pad_idx[mid_idx + 1:])), (0, 0), (0, 0), (0, 0)), mode='edge')
    return pad_frames.astype(np.uint8)


def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        x = frame['x']
        y = frame['y']
        w = frame['width']
        h = frame['height']
        if (w <= 0 or h <= 0 or
                frame['frameNumber'] == 0 or
                len(frame['Person ID']) == 0):
            continue
        framenum.append(frame['frameNumber'])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)

    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1] + 1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0, 4):
            interpfn = interp1d(framenum, bboxes[:, ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    # assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record['frameNumber'] = frame
        record['x'] = bbox[0]
        record['y'] = bbox[1]
        record['width'] = bbox[2] - bbox[0]
        record['height'] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track


def make_dataset(file_name, json_path, gt_path, stride=1,dic=None):
    logger.info('load: ' + file_name)
    # all videos
    cnt0=0
    cnt1=0
    images = []
    keyframes = []
    count = 0
    # video list
    with open(file_name, 'r') as f:
        videos = f.readlines()
    for uid in videos:
        uid = uid.strip()
        # per video
        # xxx.mp4.json
        with open(os.path.join(gt_path, uid + '.json')) as f:
            gts = json.load(f)
        positive = set()
        # load
        for gt in gts:
            for i in range(gt['start_frame'], gt['end_frame'] + 1):
                positive.add(str(i) + ":" + gt['label'])
        # json dir
        vid_json_dir = os.path.join(json_path, uid)
        # all faces
        tracklets = glob.glob(f'{vid_json_dir}/*.json')
        for t in tracklets:
            with open(t, 'r') as j:
                frames = json.load(j)
            frames.sort(key=lambda x: x['frameNumber'])
            trackid = os.path.basename(t)[:-5]
            # check the bbox, interpolate when necessary
            frames = check(frames)

            for idx, frame in enumerate(frames):
                frameid = frame['frameNumber']
                bbox = (frame['x'],
                        frame['y'],
                        frame['x'] + frame['width'],
                        frame['y'] + frame['height'])
                identifier = str(frameid) + ':' + frame['Person ID']
                label = 1 if identifier in positive else 0
                
                images.append((uid, trackid, frameid, bbox, label))
                key=uid+":"+trackid+":"+str(frameid)
                        
                if  (idx% stride == 0):
                    if dic!=None:
                        max_index=dic[key]['max_index']
                        max_score=dic[key]['array'][max_index]
                        if max_score==0:
                            count += 1
                            continue
                    
                    track_path=os.path.join('data/face_imgs',uid,trackid)
                    face_img=f'{track_path}/face_{frameid:05d}.jpg'
                    
                    if not os.path.exists(face_img):
                        count+=1
                        continue
                    keyframes.append(count)
                    if images[count][4]==1:
                        cnt1+=1
                    else:
                        cnt0+=1
                count += 1
    
    print(cnt1,cnt0)
    test_uniqueid=None
    test_frameid=None
    
    return images, keyframes,(cnt1,cnt0)

def make_test_dataset(test_path, stride=1):
    logger.info('load: ' + test_path)

    g = os.walk(test_path)
    images = []
    keyframes = []
    count = 0
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            if os.path.exists(os.path.join(test_path, dir_name)):
                uid = dir_name
                g2 = os.walk(os.path.join(test_path, uid))
                for _, track_list, _ in g2:
                    for track_id in track_list:
                        g3 = os.walk(os.path.join(test_path, uid, track_id))
                        for _, _, frame_list in g3:
                            for idx, frames in enumerate(frame_list):
                                frame_id = frames.split('_')[0]
                                unique_id = frames.split('_')[1].split('.')[0]
                                images.append((uid, track_id, unique_id, frame_id))
                                if idx % stride == 0:
                                    keyframes.append(count)
                                count += 1
    return images, keyframes



class ImagerLoader(torch.utils.data.Dataset):#148983 1024659 #170815 1028110
    def __init__(self, source_path, file_name, json_path, gt_path,
                 stride=1, scale=0, mode='train', transform=None,args=None):

        #self.source_path = source_path
        #assert os.path.exists(self.source_path), 'source path not exist'
        self.file_name = file_name
        assert os.path.exists(self.file_name), f'{mode} list not exist'
        self.json_path = json_path
        assert os.path.exists(self.json_path), 'json path not exist'
        self.source_path=source_path
        self.face_path='data/face_imgs'
        
        
        self.transform = transform
            
        
                
        if args.filter and mode=='train':
            file=open('train_filter_all.json','r')
            logger.info('Set Train Filter')
            self.dic=json.load(file)
            file.close()
        else:
            self.dic=None
            
            
            
        if args.head_query and mode=='train':
            file=open('train_headpose.json','r')
            logger.info('Load Train Headpose')
            self.dic4=json.load(file)
            file.close()
        elif args.head_query:
            file=open('val_headpose-all.json','r')
            logger.info('Load val Headpose')
            self.dic4=json.load(file)
            file.close()   
            
        images, keyframes,(lam,nlam)= make_dataset(file_name, json_path, gt_path, stride=stride,dic=self.dic)
        self.lam=lam
        self.nlam=nlam
        self.args=args
        self.imgs = images
        self.kframes = keyframes
        self.img_group = self._get_img_group()
        self.scale = scale  # box expand ratio
        self.mode = mode
    def get_labels(self):
        return self.labels
    
    def __getitem__(self, index):
        source_video = self._get_video(index)
        target = self._get_target(index)
        
        if self.args.head_query:
            uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]
            key=uid+":"+trackid+":"+str(frameid)
            yaw,pitch,roll,_,_,_=self.dic4[key]
            angle=np.floor(np.array([np.abs(yaw),np.abs(pitch),np.abs(roll)]))
            angle=torch.LongTensor(angle)
        else:
            angle=torch.zeros(1)
            
        return source_video, target,angle
    
    def _get_testvideo(self, index):
        uid, trackid, frameid, _, label= self.imgs[self.kframes[index]]
        video = []
        need_pad = False

        path = os.path.join(self.test_path, uid, trackid)
        for i in range(int(frameid) - 3, int(frameid) + 4):
            found = False
            ii = str(i).zfill(5)
            g = os.walk(path)
            for _, _, file_list in g:
                for f in file_list:
                    if ii in f:
                        img_path = os.path.join(path, f)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        video.append(np.expand_dims(img, axis=0))
                        found = True
                        break
                if not found:
                    video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                    if not need_pad:
                        need_pad = True

        video = np.concatenate(video, axis=0)
        if need_pad:
            video = pad_video(video)
        if self.args.shuffletrain and self.mode=='train':
            np.random.shuffle(video)
            
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        return video.type(torch.float32)
    
    def __len__(self):
        if self.args.flip and self.mode=='train':
            return 2*len(self.kframes)
        else:
            return len(self.kframes)

    
                
    def _get_video(self, index, debug=False):
        

        
        uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]
        video = []
        video2= []
        need_pad = False
        pad_mid=False
        key=uid+":"+trackid+":"+str(frameid)  
        
            
        for i in range(frameid - 3, frameid + 4):
            
            #index=i-frameid+3
            
            uid_path=os.path.join(self.face_path,uid)
                
            track_path=os.path.join(uid_path,trackid)
            face_img=f'{track_path}/face_{i:05d}.jpg'
                    
            #img = f'{self.source_path}/{uid}/img_{i:05d}.jpg'
            
            if i not in self.img_group[uid][trackid] or not os.path.exists(face_img):
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                if self.args.DR and self.mode=='train':
                    video2.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                if not need_pad:
                    need_pad = True
                continue
            
                
            
            assert os.path.exists(face_img), f'img: {face_img} not found'
                
            img = cv2.imread(face_img)
                
            
                    
            face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            

            if debug:
                import matplotlib.pyplot as plt
                plt.imshow(face)
                plt.show()
                
                
            video.append(np.expand_dims(face, axis=0))


        video = np.concatenate(video, axis=0)
        
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        return video.type(torch.float32)

    def _get_target(self, index):

        if self.mode == 'train':
            return self.imgs[self.kframes[index]]
        else:
            return self.imgs[self.kframes[index]]

    def _get_img_group(self):
        img_group = self._nested_dict()
        for db in self.imgs:
            img_group[db[0]][db[1]][db[2]] = db[3]
        return img_group

    def _nested_dict(self):
        return defaultdict(helper)


class TestImagerLoader(torch.utils.data.Dataset):
    def __init__(self, test_path,args,stride=1, transform=None):

        self.test_path = test_path
        assert os.path.exists(self.test_path), 'test dataset path not exist'
        
        images, keyframes = make_test_dataset(test_path, stride=stride)
        self.imgs = images
        self.kframes = keyframes
        self.transform = transform
        self.args = args
        if self.args.head_query:
            file=open('test_headpose.json','r')
            logger.info('Load Test Headpose')
            self.dic=json.load(file)
            file.close()
            
    def __getitem__(self, index):
        source_video = self._get_video(index)
        target = self._get_target(index)
        
        if self.args.head_query:
            uid, trackid, uniqueid, frameid = self.imgs[self.kframes[index]]
            key=uid+":"+trackid+":"+uniqueid+":"+str(frameid)
            yaw,pitch,roll=self.dic[key]
            angle=np.floor(np.array([np.abs(yaw),np.abs(pitch),np.abs(roll)]))
            angle=torch.LongTensor(angle)
        else:
            angle=torch.zeros(1)
        
        return source_video, target,angle

    def __len__(self):
        return len(self.kframes)

    def _get_video(self, index):
        uid, trackid, uniqueid, frameid = self.imgs[self.kframes[index]]
        video = []
        need_pad = False

        path = os.path.join(self.test_path, uid, trackid)
        for i in range(int(frameid) - 3, int(frameid) + 4):
            found = False
            ii = str(i).zfill(5)
            g = os.walk(path)
            for _, _, file_list in g:
                for f in file_list:
                    if ii in f:
                        img_path = os.path.join(path, f)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        video.append(np.expand_dims(img, axis=0))
                        found = True
                        break
                if not found:
                    video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                    if not need_pad:
                        need_pad = True

        video = np.concatenate(video, axis=0)
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        return video.type(torch.float32)

    def _get_target(self, index):
        return self.imgs[self.kframes[index]]
