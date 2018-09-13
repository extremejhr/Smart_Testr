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
    
    a. Testing more complex input and actions.\n
    b. Create the input file format class.\n
    c. Bayes classfier to correct the ocr results.\n
    
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


27.07.2018 version 0.0.0.1.x:

pywinauto - hwnwrapper.Rectangle() position coordinates. ('Simulation Navigator' - the left ribbon region)

left tap - dlg['TabControl'].Select(1)

01.08.2018 version 0.0.0.1.x:

Tooth is paining ...

New Update:

1.New OCR string comparision algrothim:
   
   1. Store the Levenshtein.ratio of one Group Region which is larger than 0.5;
   2. Find the closest group;
   3. If there are len(key_word)-1 box have more than 0.7 Levenshtein.ratio, the operation coordinates will be the center of this group.

Ibuprofen Ура!!!

2. Something need to be mentioned:

   Currently one CAE project focused on the UI Automation Test has been established. So it seems to make Smart-Testr becomes unnecessary. 
   
   In this project, UI operation can be recorded and validated in order to create UI Autotest. In other words, this project is designed for devs and these UI autotests still need testrs to record and generate.
   
   One thing need to be mentioned again: Smart-Testr is a tool designed for tester and QA. Its initial purpose is to reduce the work and increse the efficiency of testing. It tries to solve the 'Last Mile' problem in testing.
   
   Many tools have been developed for Devs to analyze and locate the error codes, but there is little tool testrs can use to find the errors and assertions. It is correct that testr's main object is to help the dev find and fix the program faults, while testing is also an important part of one software development. If one tool can help testr to test the program more efficiently, testr can take more attention on the design of test case, which could increase the testing coverage and quality. So This is why I would like to create Smart-Testr.
   
   Smart-Testr is still a little baby, there are so many drawbacks and errors in this program:Low efficiency, inacurrate OCR location, the theme color and layout have big influence... So much works need to be done. I hope Smart-Testr could have a little help to testrs in future.
   
17.08.2018 version 0.0.0.1.final:

1. Current developing process stop until the testing works finished;
2. New idea: WorkerJ - Our computer collegue/staff, AI tester. More general testing, project testing. Self-learning. Maybe more useful and innovational.
3. Peace and Love!
   
 13.09.2018 version 0.0.0.2:

1. New version development is starting!!!!!!!!yeah!!!!!!!
2. To Finish input user interface;
3. Increase code structure and efficiency.
4. To my best I think :)  
   
   
        
