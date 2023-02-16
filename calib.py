# Modified from: https://github.com/TemugeB/python_stereo_camera_calibrate
import shutil
import cv2 as cv
import glob
import numpy as np
import os

import camera.binding as camera

calibration_settings = {
    "resolution": camera.CLEyeCameraResolution.CLEYE_VGA, # 640x480, pick QVGA for 320x240
    "mono_calibration_frames": 25,
    "stereo_calibration_frames": 50,
    "assume_accurate": True, # Skips manual validation of checkerboard detection
    "checkerboard_box_size_scale" : 4, # 4cm
    "checkerboard_columns": 11,
    "checkerboard_rows": 8, 
    "cooldown": 25
}

#Open camera stream and save frames
def save_frames_single_camera(camera_name):
    #create frames directory
    if not os.path.exists('frames'):
        os.mkdir('frames')

    number_to_save = calibration_settings['mono_calibration_frames']
    cooldown_time = calibration_settings['cooldown']

    cap = camera.Camera(camera_name, camera.CLEyeCameraColorMode.CLEYE_COLOR_PROCESSED, calibration_settings["resolution"], 50)
    
    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        frame = cap.get_frame()
        frame_small = frame.copy() 

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    cv.destroyAllWindows()


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_columns']
    columns = calibration_settings['checkerboard_rows']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            if not calibration_settings['assume_accurate']:
                k = cv.waitKey(0)
        
                if k & 0xFF == ord('s'):
                    print('skipping')
                    continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


#open both cameras and take calibration frames
def save_frames_two_cams(camera0_name, camera1_name):

    #create frames directory
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    #settings for taking data
    cooldown_time = calibration_settings['cooldown']    
    number_to_save = calibration_settings['stereo_calibration_frames']

    #open the video streams
    cap0 = camera.Camera(camera0_name, camera.CLEyeCameraColorMode.CLEYE_COLOR_PROCESSED, calibration_settings["resolution"], 60)
    cap1 = camera.Camera(camera1_name, camera.CLEyeCameraColorMode.CLEYE_COLOR_PROCESSED, calibration_settings["resolution"], 60)

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True:
        frame0 = cap0.get_frame()
        frame1 = cap1.get_frame()

        frame0_small = frame0.copy()
        frame1_small = frame1.copy()

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame0)

                savename = os.path.join('frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame1)

                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    cv.destroyAllWindows()


#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_columns']
    columns = calibration_settings['checkerboard_rows']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)

            if not calibration_settings['assume_accurate']:
                k = cv.waitKey(0)

                if k & 0xFF == ord('s'):
                    print('skipping')
                    continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv.destroyAllWindows()
    return R, T

def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + '0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + '1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def get_depth(proj0, proj1, points0, points1):
    point3d = cv.triangulatePoints(proj0, proj1, points0, points1)
    point3d = cv.convertPointsFromHomogeneous(point3d.T)
    return point3d

def check_calibration(cmtx0, R0, T0, cmtx1, R1, T1):
    cam0 = camera.Camera(1, camera.CLEyeCameraColorMode.CLEYE_COLOR_PROCESSED, calibration_settings["resolution"], 60)
    cam1 = camera.Camera(0, camera.CLEyeCameraColorMode.CLEYE_COLOR_PROCESSED, calibration_settings["resolution"], 60)

    P0 = cmtx0 @ _make_homogeneous_rep_matrix(R0, T0)[:3,:]
    P1 = cmtx1 @ _make_homogeneous_rep_matrix(R1, T1)[:3,:]

    res = (640, 480) if calibration_settings["resolution"] == camera.CLEyeCameraResolution.CLEYE_VGA else (320, 240)
    optimal_cmtx0, roi0 = cv.getOptimalNewCameraMatrix(cmtx0, dist0, res, 1, res)
    optimal_cmtx1, roi1 = cv.getOptimalNewCameraMatrix(cmtx1, dist1, res, 1, res)

    while cv.waitKey(1) != 27:
        frame0 = cam0.get_frame()
        frame1 = cam1.get_frame()

        frame0 = cv.undistort(frame0, cmtx0, dist0, None, optimal_cmtx0)
        frame1 = cv.undistort(frame1, cmtx1, dist1, None, optimal_cmtx1)

        chess = (calibration_settings["checkerboard_columns"], calibration_settings["checkerboard_rows"])
        ret0, corners0 = cv.findChessboardCorners(frame0, chess, None)
        ret1, corners1 = cv.findChessboardCorners(frame1, chess, None)

        if ret0 and ret1:
            cv.drawChessboardCorners(frame0, chess, corners0, ret0)
            cv.drawChessboardCorners(frame1, chess, corners1, ret1)

            points3d = cv.triangulatePoints(P0, P1, corners0, corners1)
            points3d = cv.convertPointsFromHomogeneous(points3d.T)[0]
            depth = np.mean(points3d[:, 2])
            cv.putText(frame0, f"Depth: {depth:.2f}cm", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(frame1, f"Depth: {depth:.2f}cm", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)
    
    del cam0
    del cam1
    cv.destroyAllWindows()

def del_f():
    # delete frames folder if it exists
    if os.path.exists('frames'):
        shutil.rmtree('frames')
    # delete frames_pairs folder if it exists
    if os.path.exists('frames_pairs'):
        shutil.rmtree('frames_pairs')

if __name__ == '__main__':
    del_f()
    # delete camera_parameters folder if it exists
    if os.path.exists('camera_parameters'):
        shutil.rmtree('camera_parameters')

    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera("1") #save frames for camera0
    save_frames_single_camera("0") #save frames for camera1


    """Step2. Obtain camera intrinsic matrices and save them"""
    #camera0 intrinsics
    images_prefix = os.path.join('frames', '1*')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix) 
    save_camera_intrinsics(cmtx0, dist0, '1') #this will write cmtx and dist to disk

    #camera1 intrinsics
    images_prefix = os.path.join('frames', '0*')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx1, dist1, '0') #this will write cmtx and dist to disk


    """Step3. Save calibration frames for both cameras simultaneously"""
    save_frames_two_cams('1', '0') #save simultaneous frames


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', '1*')
    frames_prefix_c1 = os.path.join('frames_pair', '0*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Step5. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T) #this will write R and T to disk

    check_calibration(cmtx0, R0, T0, cmtx1, R, T)

    del_f()