# Smart_Testr

05.07.2018 version 0.0.0.1

UI automation test based on image recognization (focused on testing of Siemens Simcenter)

Features:
1. Keyword driven, no affected by the testing software.
2. Test programming. user defined input and output.

Drawbacks in current version:
1. Too many parameter in processing image need to be chosen in different working environment and dialogs.
2. Some keywords cannot be recognized or are recognized incorrectly.
3. Low code efficiency.
4. UI change have big influence on the recognization results.

Roadmap：
1. Finish the basic structure of smart testr engine. 
2. Smart selection of image processing parameter in order to fit in different softwares or working environment.
3. OCR training to increase the accuracy of text recognization.
4. Add template matching in program.
5. Build UI structure and icon database.
6. Refine the code structure and increase the efficiency.
7. User interface developed.

13.07.2018 version 0.0.0.1.Modified

1. New Structure.
2. Add hyperparamter class in order to select best kernel and scale parameters.
3. Need further wrapp the class.

18.07.2018 version 0.0.0.1.x

1. New code style and structure.
2. Basic class of mouse and keyboard event build.
2. Class smart_testr_engine wrapper build.
3. Add concurrent.futures multi-thread in OCR process.
4. Further increase the efficiency:
   
   After testing different multi module (group_package - have issues of incomplete ocr output), 
   the current approach is using the multi thread in OCR one group with different region images. The costing time of ocr main screen of    NX decrease from 54s to 34s, efficiency increase 38%. --- The scanning time is still too long, need to furthur optimize the code to      increase the efficiency. But for current phase, no further effort will be put on this issue. 
   
   This problem will discussed in version 0.0.0.2.
  
 5. Modify pytesseract.py --- change proc.wait() to proc.communicate() in order to avoid blocking the PIPE.
 
 6. Next step:
    
    a. Testing more complex input and actions.
    b. Create the input file format class.
    c. Bayes classfier to correct the ocr results.
