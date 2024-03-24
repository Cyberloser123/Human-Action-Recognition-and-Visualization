from mmpose.apis import (inference_topdown, init_model, inference_pose_lifter_model,
                          extract_pose_sequence, _track_by_iou, convert_keypoint_definition)
from mmpose.utils import register_all_modules, adapt_mmdet_pipeline
from mmdet.apis import inference_detector
from mmpose.structures import PoseDataSample, merge_data_samples
import cv2
import mmcv
import numpy as np
from drawer import Visu3D
from jumpDetect import JumpCounter

videoPath = "./videos/test.mp4"

BOUNDING_BOX_SCALE_FACTOR = 0.7

def processOneImage(detector, frame: np.ndarray, frameIndex: int, poseEstimator, 
                    lastResults: list, poseEstResultsList: list, nextId: int, 
                    poseLifter, visualizrFrame: np.ndarray)->(
                        dict, int, list, list, list):
    
    '''
    pipeline of this function:
        frame -> detector -> poseEstimator -> covert result into lifter format -> poseLifter
    '''
    
    poseLiftDataSet = poseLifter.cfg.test_dataloader.dataset
    poseLiftDatasetName = poseLifter.dataset_meta['dataset_name']

    # detect the person bounding boxex
    detResult = inference_detector(detector, frame)
    predInstance = detResult.pred_instances.cuda().numpy()

    bboxes = predInstance.bboxes
    bboxes = bboxes[np.logical_and(predInstance.labels == 0,
                                    predInstance.scores > 0.3)]
    
    box = bboxes
    
    # estimate pose results for current frame
    poseEstResults = inference_topdown(poseEstimator, frame, bboxes)

    _track = _track_by_iou

    poseDetDatasetName = poseEstimator.dataset_meta['dataset_name']
    poseEstResultsConverted = []


    # covert the results into the pose-lifting format
    for i, dataSample in enumerate(poseEstResults):

        predInstances = dataSample.pred_instances.cuda().numpy()
        keypoints = predInstances.keypoints

        # calculate the area and bbox
        if 'bboxes' in predInstances:
            areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                              for bbox in predInstances.bboxes])
            poseEstResults[i].pred_instances.set_field(areas, 'areas')
        else:
            areas, bboxes = [], []
            for keypoint in keypoints:
                xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                xmax = np.max(keypoint[:, 0])
                ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                ymax = np.max(keypoint[:, 1])
                areas.append((xmax - xmin) * (ymax - ymin))
                bboxes.append([xmin, ymin, xmax, ymax])
            poseEstResults[i].pred_instances.areas = np.array(areas)
            poseEstResults[i].pred_instances.bboxes = np.array(bboxes)
        
        trackId, lastResults, _ = _track(dataSample, 
                                      lastResults,
                                      0.3)
        
        if trackId == -1:
            if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                trackId = nextId
                nextId += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                keypoints[:, :, 1] = -10
                poseEstResults[i].pred_instances.set_field(
                    keypoints, 'keypoints')
                poseEstResults[i].pred_instances.set_field(
                    predInstances.bboxes * 0, 'bboxes')
                poseEstResults[i].set_field(predInstances, 'pred_instances')
                trackId = -1

        poseEstResults[i].set_field(trackId, 'track_id')
        
        # covert the keypoints for pose-lifting
        poseEstResultConverted = PoseDataSample()
        poseEstResultConverted.set_field(
            poseEstResults[i].pred_instances.clone(), 'pred_instances')
        
        poseEstResultConverted.set_field(
            poseEstResults[i].gt_instances.clone(), 'gt_instances')
        
        keypoints = convert_keypoint_definition(keypoints,
                                                poseDetDatasetName,
                                                poseLiftDatasetName)
        poseEstResultConverted.pred_instances.set_field(
            keypoints, 'keypoints')
        poseEstResultConverted.set_field(poseEstResults[i].track_id,
                                            'track_id')
        poseEstResultsConverted.append(poseEstResultConverted)

    poseEstResultsList.append(poseEstResultsConverted.copy())

    # extract the 2d sequence
    poseSeq2d = extract_pose_sequence(
        poseEstResultsList,
        frame_idx=frameIndex,
        causal=poseLiftDataSet.get('causal', False),
        seq_len=poseLiftDataSet.get('seq_len', 1),
        step=poseLiftDataSet.get('seq_step', 1)
    )

    # 2d to 3d pose-lifting
    poseLiftResults = inference_pose_lifter_model(
        poseLifter,
        poseSeq2d,
        image_size=visualizrFrame.shape[:2],
        norm_pose_2d=True
    )

    # pose-processing
    for idx, poseLiftResult in enumerate(poseLiftResults):
        poseLiftResult.track_id = poseEstResults[idx].get('track_id', 1e4)

        pred_instances = poseLiftResult.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            poseLiftResults[
                idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = keypoints[..., [0, 2, 1]]
        keypoints[..., 0] = -keypoints[..., 0]
        keypoints[..., 2] = -keypoints[..., 2]

        
        keypoints[..., 2] -= np.min(
            keypoints[..., 2], axis=-1, keepdims=True)

        poseLiftResults[idx].pred_instances.keypoints = keypoints

    poseLiftResults = sorted(
        poseLiftResults, key=lambda x: x.get('track_id', 1e4))
    
    pred3dDataSamples = merge_data_samples(poseLiftResults)
    pred3dInstances = pred3dDataSamples.get('pred_instances', None)

    return poseEstResults, poseEstResultsList, pred3dInstances, nextId, box
        
def visualization(predInstances:list, movement:list):

    v3d = Visu3D()
    v3d.setVerticalMovement(movement)
    v3d.addGeometry(predInstances)
    v3d.run()

def scaleBox(box, f):
    if box is None:
        return None
    (x, y, w, h) = box
    return x + int(.5 * w * (1 - f)), y + int(.5 * h * (1 - f)), w * f, h * f

def biggerBox(box):
    return scaleBox(box, 0.9/BOUNDING_BOX_SCALE_FACTOR)

def main():
    register_all_modules()

    # load model
    configFile = './models/rtmdet_m_640-8xb32_coco-person.py'
    checkpointFile = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    detector = init_model(configFile, checkpointFile, device='cuda:0')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    configFile = './models/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpointFile = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    poseEstimator = init_model(configFile, checkpointFile, device='cuda:0')

    configFile = './models/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py'
    checkpointFile = 'https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth'
    poseLifter = init_model(configFile, checkpointFile, device='cuda:0')
    
    # get video
    video = cv2.VideoCapture(videoPath)

    # variable for recording detection result
    nextId = 0
    poseEstResultsList = []
    poseEstResults = []
    predInstances = []
    frameIndex = 0
    jumpDector = JumpCounter()
    jumpDector.setPersonHeight(1.7)

    while video.isOpened():
        ret, frame = video.read()
        frameIndex += 1
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC)

        if not ret:
            break  
        lastEstResults = poseEstResults
        # process every frame
        (poseEstResults, poseEstResultsList, pred3dInstances, nextId, box) = processOneImage(
            detector=detector,
            frame=frame,
            frameIndex=frameIndex,
            poseEstimator=poseEstimator,
            lastResults=lastEstResults,
            poseEstResultsList=poseEstResultsList,
            nextId=nextId,
            poseLifter=poseLifter,
            visualizrFrame=mmcv.bgr2rgb(frame),
        )
        
        box[0][2] = box[0][2] - box[0][0]
        box[0][3] = box[0][3] - box[0][1]

        jumpDector.countJumps(biggerBox(box[0]), timestamp)  
              
        predInstances.append(pred3dInstances.get('keypoints'))

    result = jumpDector.getJumpHeight()
    
    video.release()
    
    # 3D pose visualization
    visualization(predInstances, result)
    

if __name__ == '__main__':
    main()
    