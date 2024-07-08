# UIDemoStream
Youtube UI DEMO || https://youtu.be/TTB5y-03SnE?list=PLUJEyvnin-qMKP4gs-oZ9bMl4-1INEVIi<br/>
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

2. Setup data table for input actions (sample)<br/>
  * Gamepad Input Overrides는 다른 입력 기준을 가진 플랫폼 대응하기 위함.
  * UI/Icons 폴더에 Icon 추가
  * 모든 아이콘 선택 후, [Common] - [Asset Actions] - Bulk Edit via Property Matrix 
    * LevelOfDetail - Texture Group - UI선택
    * InputData_PC_OOO - Input Brush Data Map - Image에 아이콘 넣기 사이즈 32x32
  
|위치|DisplayName|RowName|keyboard|gamepad|
|:---:|:---:|:---:|:---:|:---:|
|InputActionTable - Add - Display|Confirm|Confirm|Enter|gamepad face button bottom|
|InputActionTable - Add - Display|Cancel|Cancel|Escape|gamepad face button right|
|InputActionTable - Add - Display|Tableft|Tableft|q|gamepad left shoulder|
|InputActionTable - Add - Display|TabRight|TabRight|r|gamepad right shoulder|

3. Generic data input blueprints hold info about button icons <br/>
  * UI/Data 경로 [Create basic asset] - [Blueprint Class] - CommonInputBaseControllerData  "ControllerData_PC_keyboard"
  * UI/Data 경로 [Create basic asset] - [Blueprint Class] - CommonInputBaseControllerData  "ControllerData_PC_gamepad"
  * 밑에 표와 같이 작성
  * Input Brush Data Map - Add Element 4번
  * Icons폴더

|위치|[Details] - Default - Input Type|[Details] - Gamepad - Gamepad Name|[Details] - Gamepad - Gamepad Display Name|[Details] - Gamepad - Gamepad Platform Name|
|:---:|:---:|:---:|:---:|:---:|
|ControllerData_PC_keyboard|Mouse and keyboard|NA|NA|NA|
|ControllerData_PC_gamepad|Gamepad|Generic|Gamepad|Windows|

4. Common input data blueprint <br/>
 * UI/Data - [Create basic asset] - [Blueprint Class] - CommonUIInputData "DemoGameInputData"
   * 뒤로가기와 클릭 설정
     * DataTable과 RowName 설정(Confirm, Cancel)
       * DataTable: InputActionTable
       * RowName: Confirm, Cancel 선택 
 * [Project Settings] - [Game] - Common Input Settings
   * Input - Input Data: DemoGameInputData
   * Platform Input - Windows
     * Controller Data
       * ControllerData_PC_keyboard
       * ControllerData_PC_gamepad  

5. styling assets <br/>
  * UI/Data 경로 Styling 폴더 추가
    *  [Create basic asset] - [Blueprint Class] - CommonBorderStyle "BorderStyle_DemoGameGenericBorder"
    *  [Create basic asset] - [Blueprint Class] - CommonButtonStyle "ButtonStyle_DemoGameGenericButton"
    *  [Create basic asset] - [Blueprint Class] - CommonTextStyle "TextStyle_DemoGameGenericMenuText"
  * Content drawer에서 Settings - Show enging content 체크하면 글꼴 보임
  * 표준 UMG 설명 ... [create advanced asset] - [User Interface] TestUser

6. UI
   * [Settings] - Plugins common ui editor에 블루프린트 클래스추가

7. Don't use the canvas panel <br/>
  * canvas는 연산량이 많아서 최소한으로 사용해야 한다.
  * overlay 사용하면 여러 요소들을 하위에 둘 수 있다.
  * UI폴더 [create basic asset] - [blueprint class] - CommonActivatableWidget "UI_Base"
  * overlay
    * Common Activatable Widget Stack(build menu stack) "MenuStack"
      * Menustack fill horizontally, fill vertically
      * common activatable widget stack "PromptStack"
        * [Graph] - add custom event "PushMenu"
        * [Graph] - add custom event "PushPrompt"
  * que에 대해서도 공부

8. StackCommonWidget <br/>
  * 
