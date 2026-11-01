; Flyweight Desktop installer for Windows (Inno Setup 6).
;
; Built by tools/build_desktop.py --installer, which passes:
;   /DAppVersion=0.26.0  /DSourceDir=<the PyInstaller bundle>  /DOutputDir=<dist>
;
; A per-user install: no administrator prompt, nothing outside the user's own
; folders, and an entry in Settings > Apps for uninstalling.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Flyweight"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{6F1C2B7E-3D9A-4E52-9B57-0F4E1C7A9D21}
AppName=Flyweight
AppVersion={#AppVersion}
AppVerName=Flyweight {#AppVersion}
AppPublisher=Flyweight
AppPublisherURL=https://github.com/yairpatch/flyweight
AppSupportURL=https://github.com/yairpatch/flyweight/issues
AppUpdatesURL=https://github.com/yairpatch/flyweight/releases
DefaultDirName={localappdata}\Programs\Flyweight
DefaultGroupName=Flyweight
DisableProgramGroupPage=yes
DisableDirPage=auto
DisableReadyPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=Flyweight-Setup-windows-x86_64
SetupIconFile=..\icons\flyweight.ico
UninstallDisplayIcon={app}\Flyweight.exe
UninstallDisplayName=Flyweight
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; A running Flyweight holds its files open; offer to close it on upgrade.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; An upgrade replaces the bundle wholesale: stale modules from an older
; version must not shadow the new ones.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\Flyweight"; Filename: "{app}\Flyweight.exe"; Comment: "Run AI models on this computer"
Name: "{autodesktop}\Flyweight"; Filename: "{app}\Flyweight.exe"; Tasks: desktopicon; Comment: "Run AI models on this computer"

[Run]
Filename: "{app}\Flyweight.exe"; Description: "Launch Flyweight"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The GPU support Flyweight downloaded (in %LOCALAPPDATA%\Flyweight). Settings
; in %APPDATA%\flyweight and the user's models are left alone.
Type: filesandordirs; Name: "{localappdata}\Flyweight"
