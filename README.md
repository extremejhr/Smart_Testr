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
