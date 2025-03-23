### Controls
- c: Go to next image
- q: Close the entire program
- r: Delete current (red) segmentation and start over
- spacebar: Accept current (red) segmentation (turns it green)
- left and right arrow keys: Alternate between SAM proposed masks
- left mouse click: Add clicked point to set of positive points for SAM
- right mouse click: Add clicked point to set of negative points for SAM (makes it so that the clicked area will not be segmented hopefully)
### Notes
- Adding many points to SAM doesn't always give good results, if a log is troublesome, it might be better to press 'r' and start over   
- All necessary config should be on lines 28 to 32    
- Only way to delete accepted (green) segmentations is to manually delete the corresponding bitmaps from segmentations/ folder
- Before running you need to create the src/segmentations folder
- To install just run install.sh from the folder that it is in. Append ' 2>&1 | tee err_log.txt' to get script log to fi.e.
- sam_annotate.py needs to be run from inside src folder, because if the folder that sam_annotate.py is in also has the sam2 folder, then it complains that the local folder might conflict with installed version of sam2