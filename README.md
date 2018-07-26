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
    
21.07.2018 version 0.0.0.1.x

1. Delete the box coordinates function and hyper-parameter for decreasing the infulence of different resolution.
2. Add new function of grouping region coordinates. The efficiency decrease slightly but more stable.
3. Add Tesseract Training class for capturing the icon samples. (Considering for the new traindata creation)

4. Next Step:

    a. Add try and exception to capture any error and exception situation.
    b. Add icon matchTemplate in opencv and action record.
    
25.07.2018 version 0.0.0.1.x

New idea:

1. After mouse and keyboard operation, detect and only OCR the difference image between previous and current window capture image.
2. Text and its nearest icon synchronization. if text maching fail, conduct icon template match.
3. Modify the image capture strategy, no need to specify the operation window title.
4. Apply more easy and obvious way to design the operation sequence, maybe in a GUI. Tools no clues now.

Questions need to be answered: is it easy to use? does it speed up the testing process?
    
26.07.2018 version 0.0.0.1.x

pywinauto - class Windowspecification: 

app =  Application().connect(process = ''/title_re = '')

dlg = app.window(title = '')

criteria = dlg.criteria

this_ctrl=dlg._WindowSpecification__resolve_control(dlg.criteria,)[-1]

all_ctrls = [this_ctrl, ] + this_ctrl.descendants()

for ctrl in all_ctrls:
    
    if len(ctrl.window_text())>0:
    
        print(ctrl.window_text())

Create inheritance class of pywinauto.application.WindowSpecification in order to get the text information.
        
