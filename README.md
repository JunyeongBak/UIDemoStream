# UIDemoStream
Youtube UI DEMO || https://youtu.be/TTB5y-03SnE?list=PLUJEyvnin-qMKP4gs-oZ9bMl4-1INEVIi<br/>

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


## 1. Required engine settings setup <br/>
  * TopDown프로젝트 C++로 시작
  * [Edit] - [Plugins] - Common UI Plugin : 활성화
  * [Edit] - [Project Settings] - Game Viewport Client Class - CommonGameViewportClient 선택
  * ContentDrawer에서 {Maps, Blueprint, UI} 폴더 추가
  * Maps폴더 DemoGameFrontend level 추가
  * UI폴더 {Data, Icons} 폴더 추가
  * UI/Data 경로에 [Create advanced] - [asset Miscellaneous] - [Data table] - CommonInputActionDataBase 선택하고 "InputActionTable" 추가(UI에서 인식할 수 있는 모든 입력 작업을 포함하는 입력 작업 데이터)

## 2. Setup data table for input actions (sample)<br/>
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

## 3. Generic data input blueprints hold info about button icons <br/>
  * UI/Data 경로 [Create basic asset] - [Blueprint Class] - CommonInputBaseControllerData  "ControllerData_PC_keyboard"
  * UI/Data 경로 [Create basic asset] - [Blueprint Class] - CommonInputBaseControllerData  "ControllerData_PC_gamepad"
  * 밑에 표와 같이 작성
  * Input Brush Data Map - Add Element 4번
  * Icons폴더

|위치|[Details] - Default - Input Type|[Details] - Gamepad - Gamepad Name|[Details] - Gamepad - Gamepad Display Name|[Details] - Gamepad - Gamepad Platform Name|
|:---:|:---:|:---:|:---:|:---:|
|ControllerData_PC_keyboard|Mouse and keyboard|NA|NA|NA|
|ControllerData_PC_gamepad|Gamepad|Generic|Gamepad|Windows|

## 4. Common input data blueprint <br/>
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

## 5. styling assets <br/>
  * UI/Data 경로 Styling 폴더 추가
    *  [Create basic asset] - [Blueprint Class] - CommonBorderStyle "BorderStyle_DemoGameGenericBorder"
      * [Background] - Tint 색 변경(Hex linear 090502FF, Hex sRGB 362815FF)   
    *  [Create basic asset] - [Blueprint Class] - CommonButtonStyle "ButtonStyle_DemoGameGenericButton"
      * 텍스트 등록 시 c++ 수준이나 그와 유사한 수준(blueprint?)에서 등록해줘야한다. 그러므로 지금은 builtin 스타일로 진행
      * 아래 컬러 등록은 연습용이라서 사용! 실제로는 이미지로 함.
      * [Normal Base] - Tint 색 변경 (Hex Linear 2E2E1FFF, Hex sRGB7676662FF)
      * [Normal Hovered] - Tint 색 변경 (Hex Linear FFC400FF, Hex sRGB FFE300FF)
      * [Normal Pressed] - Tint 색 변경 (Hex Linear 010108FF, Hex sRGB 090A31FF)
      * [Disabled] - Tint 색 변경(Hex Linear 1B1B1BFF, Hex sRGB 5B5B5BFF)
    *  [Create basic asset] - [Blueprint Class] - CommonTextStyle "TextStyle_DemoGameGenericMenuText"
  * Content drawer에서 Settings - Show enging content 체크하면 글꼴 보임
  * 표준 UMG 설명 ... [create advanced asset] - [User Interface] TestUser

## 6. UI
  * [Settings] - Plugins common ui editor에 블루프린트 클래스추가

## 7. Don't use the canvas panel <br/>
## 8. StackCommonWidget <br/>
  * canvas는 연산량이 많아서 최소한으로 사용해야 한다.
  * overlay 사용하면 여러 요소들을 하위에 둘 수 있다.
  * UI폴더 [create basic asset] - [blueprint class] - CommonActivatableWidget "UI_Base"
  * overlay
    * Common Activatable Widget Stack(build menu stack) "MenuStack"
      * Menustack fill horizontally, fill vertically
      * [Graph] - 우클릭 add custom event "PushMenu"
      * [Graph] - 우클릭 add custom event "PushPrompt"
      * 그래프 52분 55초
        * VARIABLES - MenuStack - GetMenuStack
          * 드래그 push widget추가
          * PushMenu 우측화살표와 pushwidget 좌측화살표 연결
          * PushMenu 《activatable Widget Class》 push widget
    * common activatable widget stack "PromptStack" 팝업 모달창 연관
      * VARIABLES - PromptStack - GetMenuStack
        * 드래그 push widget추가
        * PushPrompt 우측화살표와 pushwidget 좌측화살표 연결
    * common activatable widget queue에 대해서도 공부

## 9. CommonButtonBase <br/>
  * [CREATE BASIC ASSET] - [Blueprint Class] CommonActivatableWidget "UI_MainMenu"
  * [CREATE BASIC ASSET] - [Blueprint Class] CommonButtonBase "UI_Generic_Button"
    * Overlay 추가
    * [UI_Generic_Button] - [Details] - [Style] - ButtonStyle_DemoGameGenericButton
    * 뷰포트 우측 상단 Width 250, Height 60
    * Common Text 추가 가운데 정렬
    * Is Variable 체크하기 "DisplayedText"
      * 드래그 위젯의 set Text 추가
        * Event Pre Construct 우측화살표와 SetText 좌측 화살표 연결
        * DisplayedText와 Target 연결
        * ButtonText와 In Text 연결
    * [Graph] - [VARIABLES] - Text 추가 "ButtonText"
      * VARIABLES의 Name 상수 문자열
      * VARIABLES의 String 동적 문자열
      * VARIABLES의 Text localize텍스트, [Tools] - Localization Dashboard
      * Default Value 쪽의 플래그 눌러서 localize 셋팅한다.
    * [Selection] - selectable은 라디오버튼이나 체크박스 사용 할 때 쓴다.

## 10. MainMenu <br/>
  * [UI_MainMenu] - Overlay 추가
    * Vertical Box 추가(vertical fill 적용, bottom padding 추가) 
      * spacer 추가 (상단 padding) 
      * UI_Generic_Button 추가 "NewGameButton"
      * UI_Generic_Button 추가 "ContinueButton"
      * UI_Generic_Button 추가 "OptionsButton"
      * Spacer 추가 후 size fill 적용
      * UI_Generic_Button 추가 "QuitGameButton"
      * spacer 추가 (하단 padding)
    * [Vertical Box] - [Wrap With..] - Common Border

## 11. PlayerController <br/>
  * [Blueprint] 폴더 - [Create Basic asset] - [Blueprint Class] - [Player Controller] "FrontEndPlayerController"
    * [Event Graph] - [Create Widget]
      * class: UI Base 선택
      * owning Player: self
      * return value 드래그 Add to Viewport 추가
        * create widget 우측 화살표를 좌측에 연결
        * 우측 화살표를 드래그 push menu 추가(좌측에 연결)
          * Activatable Widget class: UI Main Menu선택
  * [Blueprint] 폴더 - [Create Basic asset] - [Blueprint Class] - [Game Mode Base]
    * [Event Graph] - [Classes] - [PlayerController Class] FrontendPlayerController 선택
  * [World Settings] - [Selected GameMode] FrontEndGameMode 선택
  
## 12. MainMenu (Graph) <br/>
  * [Event On Activated] 추가
    * 우측화살표 드래그, Set Focus 추가
      * Get Desired Focus Target 추가, Target에 연결
        * [FUNCTIONS] - Get Desired Focus Target(override)
          * [VARIABLES] NewGameButton을 Return Value에 연결
    * [ProjectSettings] - [CommonInputSettings] - Default Input Type을 gamepad로 하게되면, 기본으로 호버링 기능이 작동.
      * 다만, 키보드로 메뉴를 조작하는 경우에, 호버링이 작동하지 않는다.(Custom Event 만든 후 UpdateButtonHoverState???)

## 13. Create Generic Prompt <br/>
  * 레이아웃은 좋은 레퍼런스를 찾아야한다.
  * [UI] 폴더 - [Create Basic asset] - [Blueprint Class] - [CommonActivatableWidget] 추가 "UI_GenericPrompt"
    * Overlay
      * Vertical Box(H,V 중앙정렬, v fill)
        * Common Border (Style - BorderStyle_DemoGameGenericBorder 확인)
          * Vertical Box(버티컬 중앙에 모달 창이 오도록 레이아웃)
            * Common Text "PromptText"
            * Horizontal Box
              * UI Generic Button "YesButton"
              * Spacer
              * UI Generic Button "NoButton"
    * [Graph] 작업
      * [Function] - Get Desired Focus Target 
        * YesButton 추가, Return Value 연결
      * Event On Activated 추가
        * Get DesiredFocus Target, Set Focus 추가
          * Event On Activated 우측화살표를 Set Focus 좌측에 연결
          * Get Desired Focus Target을 Set Focus Target에 연결
      * Add Custom Event 추가 "SetPromptInfo"
        * Inputs 추가 "InPromptText" Text Type
        * delegate binding하면, c++ code에서 편리하게 할 수 있다고 합니다.

### HTML 
  * Character	Escape Code	Description
&	&amp;	Ampersand
<	&lt;	Less-than sign
>	&gt;	Greater-than sign
"	&quot;	Double quotation mark
'	&apos;	Single quotation mark
©	&copy;	Copyright symbol
®	&reg;	Registered trademark
€	&euro;	Euro sign
£	&pound;	Pound sign
¥	&yen;	Yen sign
공백 	&nbsp;	Non-breaking space
