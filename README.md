# UIDemoStream
Youtube UI DEMO || https://youtu.be/TTB5y-03SnE?list=PLUJEyvnin-qMKP4gs-oZ9bMl4-1INEVIi 
UE AI with Behavior || Trees https://www.youtube.com/watch?v= iY1jnFvHgbE

17:30 Required engine settings setup <br/>
21:00 Setup data table for input actions <br/>
26:20 Generic data input blueprints hold info about button icons <br/>
36:00 Common input data blueprint <br/>
39:00 styling assets <br/>
45:00 UI <br/>
48:00: Don't use the canvas panel <br/>
50:00 StackCommonWidget <br/>
53:00 CommonButtonBase <br/>
1:10:00 MainMenu <br/>
1:16:00 PlayerController <br/>
1:31:00 PromptMenu <br/>


1. Required engine settings setup <br/>
  * TopDown프로젝트 C++로 시작
  * [Edit] - [Plugins] - Common UI Plugin : 활성화
  * [Edit] - [Project Settings] - Game Viewport Client Class - CommonGameViewportClient 선택
  * ContentDrawer에서 {Maps, Blueprint, UI} 폴더 추가
  * Maps폴더 DemoGameFrontend level 추가
  * UI폴더 {Data, Icons} 폴더 추가
  * UI/Data 경로에 [Create advanced] - [asset Miscellaneous] - [Data table] - CommonInputActionDataBase 선택하고 "InputActionTable" 추가(UI에서 인식할 수 있는 모든 입력 작업을 포함하는 입력 작업 데이터)
  * 

2. Setup data table for input actions (sample)<br/>
  * Gamepad Input Overrides는 다른 입력 기준을 가진 플랫폼 대응하기 위함.
  
|위치|DisplayName|RowName|keyboard|gamepad|
|:---:|:---:|:---:|:---:|:---:|
|InputActionTable - Add - Displa|Confirm|Confirm|Enter|gamepad face button bottom|
|InputActionTable - Add - Displa|Cancel|Cancel|Escape|gamepad face button right|
|InputActionTable - Add - Displa|Tableft|Tableft|a|gamepad left shoulder|
|InputActionTable - Add - Displa|TabRight|TabRight|d|gamepad right shoulder|
