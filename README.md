# ToucanTrack
ToucanTrack is an app which achieves Full Body Tracking for VRChat using 2 PS3 Eye Cameras and AI.
> **Note**: ToucanTrack is still in beta! Expect instability and issues.  

Demo: https://streamable.com/c7zstj

### Installation
Cloning the github library and installing the required libraries:
```bash
git clone https://github.com/noahcoolboy/toucan-track.git
cd toucan-track
pip install python-osc numpy opencv-contrib-python scipy onnxruntime pyjson5
```

Follow these instructions for downloading the PS3 Eye Camera drivers: https://github.com/opentrack/opentrack/wiki/PS3-Eye-open-driver-instructions


### Calibration
For ToucanTrack to function properly, you will have to calibrate both cameras.  
Print out a checkerboard pattern as big as possible, like on an A2 paper. (Eg. [9x12 Checkerboard (20mm)](https://github.com/noahcoolboy/toucan-track/files/10776556/checker_297x210_9x12_20.pdf))  
Run `calibtool.py` and pop open the settings. If you're using a different checkerboard size, modify `Checkerboard Columns` and `Checkerboard Rows`. Measure the size of a checkerboard square, and set `Checkerboard Box Size`. Make sure all settings match the paper you have printed out correctly.
> **Note**: Because of how OpenCV works, `Checkerboard Columns` and `Checkerboard Rows` do not count the amount of squares, but the amount of corners. This means that the value should be one less than the number of squares in each direction. For the example pattern listed above, this would be 11 columns and 8 rows.

Once you're done with configuring the tool, click on save. Press `Camera +` to add a camera. Pick the type of camera, and the ID of the camera. You might have to guess the ID when having more than 2 cameras. Now select the camera in the list on the left and press `Calibrate Intrinsics`. Place the calibration pattern as close as possible (while still being fully visible) to the camera, the frames will be collected automatically. You can track your progress by looking at the text underneat the camera preview. A good calibration image should look like the following:  
<img src="https://user-images.githubusercontent.com/46800081/219941473-32608127-87e7-4a2d-accd-9b0df8b03f18.png" width=300>

Once the frames have been collected, repeat this process with the other camera. After that, the camera distortion for the cameras has been calculated. The previews should show the undistorted preview. You can now move on to the extrinsics calibration.

Print out an aruco marker (Eg. [ID 0 (18x18cm)](https://user-images.githubusercontent.com/46800081/219941888-1968b0d6-c23a-4d25-bc70-681931375418.svg)) and place it in the middle of the room on a flat surface. Measure the size of the aruco marker in centimeters, and modify the `Aruco Size` setting. Press `Calibrate Extrinsics`, this will make a window appear showing all the cameras. Cameras which can see the aruco marker will be tinted green. Make sure all cameras show green before pressing `Calibrate`. Make sure the blue line on the first camera points towards the direction you would normally face while playing VRChat. It does not matter where the aruco marker is, but it does matter how it is oriented. An example of this calibration:  
<img src="https://user-images.githubusercontent.com/46800081/219943106-4e0e4fa8-2074-4eb8-b619-1a87fc24f83a.png" width=300>  

You can now press `Save` and exit the calibration tool.

### Usage
Now that both your cameras are ready and set up, you can start setting up the main app. Open `settings.json` in any text editor. The first option you'll have to change is the IP. This should be the IP address of your Quest 2 connected on your WiFi network. The debug mode is on by default, but if everything is working well, you can turn it off. By scrolling down you can find the filter settings, which can be modified for having a smoother but laggier or more responsive but jittery experience.

When you're done fine tuning the settings, run `python main.py`. This will run ToucanTrack. You can now hop into your VRChat. You should see a `Calibrate FBT` button in your menu. Stand up straight and look forward, and press both triggers. And you're done!


### Tips
* Unable to get a good intrinsic calibration?
  * Stick the calibration pattern on a flat surface (like cardboard)! Holding the paper by hand will cause calibration issues.
  * Increase `Mono Calibration Frames`. This will take more pictures of the calibration pattern, and therefore increase calibration accuracy.
  * Make sure the calibration pattern isn't too far or close to the camera, the checkerboard won't be detected otherwise!
  * Try putting the calibration pattern at different distances and angles from the camera.
* Unable to get a good origin calibration?
  * Make sure the aruco marker is perfectly flat (on the ground or on a flat surface).
  * Put the aruco marker as close as possible to the cameras, while still staying perfectly flat. (Eg. [like this](https://user-images.githubusercontent.com/46800081/220600125-41898c07-ae69-418e-b9ae-4a79d0f7e601.png))
  * Make sure the blue line points towards the direction you would normally face when playing VRChat.
  * Make sure to measure and configure `Aruco Size` correctly!
* Bad tracking?
  * Bad lighting might cause problems. Try turning on your room's light's and closing the windows (to avoid lighting from outside).
  * Wear clothes which do not blend with the background and stay tight on the skin. Loose clothing might throw off the tracking.
  * Adjust the settings for better keypoint smoothing. The default settings do a pretty good job at avoiding jittering.
  * Verify your calibration by setting `draw_pose` to true, and facing the camera directly. Make sure the skeleton in the pose faces the southeast direction ([like this](https://user-images.githubusercontent.com/46800081/220957758-152a0cca-a5df-49da-afd9-11cd2503a369.png)). If the skeleton is not facing the correct direction or upside down, modify `flip_x`, `flip_y` and `flip_z` in the settings. If the skeleton is bent or not standing up straight, something has gone wrong during calibration.
* Trackers go too far / not far enough?
  * Make sure you have configured `Checkerboard Box Size` and `Aruco Size` correctly during the calibration process. Remember the values should be in centimeters, and not inches.
  * Configure `scale_multiplier` in the settings.
* Other problems? Contact me on discord (noah#8315) or on reddit (u/Noahcoolbot) and I will gladly help.