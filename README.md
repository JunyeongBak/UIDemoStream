# UIDemoStream
Youtube UI DEMO || https://youtu.be/TTB5y-03SnE?list=PLUJEyvnin-qMKP4gs-oZ9bMl4-1INEVIi 

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
  * [Edit] - [Project Settings] - Game Viewport Client Class : CommonGameViewportClient 선택
  * ContentDrawer에서 {Maps, Blueprint, UI} 폴더 추가
  * Maps폴더 DemoGameFrontend level 추가
  * UI폴더 {Data, Icons} 폴더 추가
  * UI/Data 경로에 [Create advanced] - [asset Miscellaneous] - [Data table] - CommonInputActionDataBase 선택하고 "InputActionTable" 추가(UI에서 인식할 수 있는 모든 입력 작업을 포함하는 입력 작업 데이터)
  * 
