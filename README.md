# ToucanTrack
ToucanTrack is an app which achieves Full Body Tracking for VRChat using 2 PS3 Eye Cameras and AI.
> **Note**: ToucanTrack is still in beta! Expect instability and issues.

### Installation
Cloning the github library and installing the required libraries:
```bash
git clone https://github.com/noahcoolboy/toucan-track.git
cd toucan-track
pip install pythonosc numpy opencv-contrib-python scipy onnxruntime pyjson5
```

Install the [CL-Eye Driver](https://drive.google.com/uc?export=download&id=1O8yER02vQ-PgeF20N0nfid1GeEduyhvk) and [CL-Eye Platform SDK](https://drive.google.com/uc?export=download&id=1OIBxT0Q9KgjaOROA5y0bYfwK92jtEfTz).  
Make sure no other PS3 Camera Driver is currently installed on your computer.


### Calibration
For ToucanTrack to function properly, you will have to calibrate both cameras.  
Print out a checkerboard pattern as big as possible, like on an A2 paper. (Eg. [9x12 Checkerboard (20mm)](https://github.com/noahcoolboy/toucan-track/files/10776556/checker_297x210_9x12_20.pdf))  
If you're using a different checkboard size, open up `calib.py` in your favorite code / text editor. At the top of the file, you'll see `calibration_settings`. Make sure that `checkerboard_columns`, `checkerboard_rows`, and `checkerboard_box_size_scale` match the checkerboard you printed out.
> **Note**: Because of how OpenCV works, `checkerboard_columns` and `checkerboard_rows` do not count the amount of squares, but the amount of corners. This means that the value should be one less than the number of squares in each direction. For the example pattern listed above, this would be 11 columns and 8 rows.

Once you're done modifying `calib.py`, you can run it using `python calib.py`.
A window will appear showing one of your connected PS3 cameras. Place the calibration pattern as close as possible (while still being fully visible) to the camera, and press space to start the collection of frames. A good calibration image should look like the following:  
<img src="https://user-images.githubusercontent.com/46800081/219941473-32608127-87e7-4a2d-accd-9b0df8b03f18.png" width=300>

Once the frames have been collected, repeat this process with the other camera. After that, the camera distortion will be calculate for both cameras, this may take a minute. When it's done, both cameras will be shown. This part is where camera extrinsics will be calibrated. Make sure both cameras can see the calibration pattern, and press space to start the collection of frames. While the frames are being collected, move the pattern around but keep it in frame for both cameras. After having collected all the frames, the camera extrinsics will be calculated. Once done, both cameras will pop up again, and you can verify the calibration results. By placing the calibration pattern infront of both cameras again, the estimated depth will be displayed in centimeters.

For improving accuracy, you can set the world origin. Print out an aruco marker (Eg. [ID 0 (18x18cm)](https://user-images.githubusercontent.com/46800081/219941888-1968b0d6-c23a-4d25-bc70-681931375418.svg)) and place it in the middle of the room. Run `python calib.py origin`. This show the detected aruco marker from your camera feed. Make sure the Z (blue) axis is pointing towards the main direction of your playspace (the way which you face most of the time while playing). An example of this calibration:  
<img src="https://user-images.githubusercontent.com/46800081/219943106-4e0e4fa8-2074-4eb8-b619-1a87fc24f83a.png" width=300>  
Press enter if the world origin looks good and well aligned, otherwise press any other key. The console will say `Successfully set origin!` if all went well.

### Usage
Now that both your cameras are ready and set up, you can start setting up the main app. Open `settings.json` in any text editor. The first option you'll have to change is the IP. This should be the IP address of your Quest 2 connected on your WiFi network. The debug mode is on by default, but if everything is working well, you can turn it off. By scrolling down you can find the filter settings, which can be modified for having a smoother but laggier or more responsive but jittery experience.

When you're done fine tuning the settings, run `python main.py`. This will run ToucanTrack. You can now hop into your VRChat. You should see a `Calibrate FBT` button in your menu. Stand up straight and look forward, and press both triggers. And you're done!